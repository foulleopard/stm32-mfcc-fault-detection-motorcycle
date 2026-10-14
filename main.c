/* ============================================================
 *  Motorcycle Fault Detection System
 *  STM32F411 Black Pill
 *
 *  Hardware:
 *    INMP441 I2S Microphone  -> I2S2 (PB12=WS, PB13=CLK, PB15=SD)
 *    MPU6050 Accelerometer   -> I2C1 (PB6=SCL, PB7=SDA)  [future]
 *    FAULT LED               -> PB1  (HIGH = fault detected)
 *    OK    LED               -> PB2  (HIGH = no fault / healthy)
 *    Trigger Button          -> PA0  (onboard USER button, active LOW)
 *
 *  Flow:
 *    PA0 press -> 5-second I2S DMA capture -> Preprocessing ->
 *    FFT -> Mel Filterbank -> MFCC (13 coeff) ->
 *    Statistical Aggregation -> Feature Vector (29-dim) ->
 *    Cosine Similarity + Pearson vs Reference Vector ->
 *    PB1 / PB2 LED toggle
 * ============================================================ */

#include "main.h"
#include "arm_math.h"       /* CMSIS-DSP */
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ============================================================
 *  CONFIGURATION DEFINES
 * ============================================================ */
#define SAMPLE_RATE         16000U   /* Hz  — INMP441 configured to 16 kHz  */
#define RECORD_SECONDS      5U
#define TOTAL_SAMPLES       (SAMPLE_RATE * RECORD_SECONDS)   /* 80 000 */

#define FRAME_SIZE          512U     /* FFT points (power-of-2)             */
#define HOP_SIZE            256U     /* 50 % overlap                        */
#define N_MELS              26U
#define N_MFCC              13U
#define FEAT_VEC_LEN        29U      /* 13 mean + 13 std + RMS + ZCR + Kurt */

#define COS_THRESHOLD       0.92f    /* below = FAULT                       */
#define PEARSON_THRESHOLD   0.92f

#define LED_FAULT_PIN       GPIO_PIN_1   /* PB1 */
#define LED_OK_PIN          GPIO_PIN_2   /* PB2 */
#define LED_GPIO_PORT       GPIOB

#define FAULT_LED_ON()      HAL_GPIO_WritePin(LED_GPIO_PORT, LED_FAULT_PIN, GPIO_PIN_SET);   \
                            HAL_GPIO_WritePin(LED_GPIO_PORT, LED_OK_PIN,    GPIO_PIN_RESET)
#define OK_LED_ON()         HAL_GPIO_WritePin(LED_GPIO_PORT, LED_OK_PIN,    GPIO_PIN_SET);   \
                            HAL_GPIO_WritePin(LED_GPIO_PORT, LED_FAULT_PIN, GPIO_PIN_RESET)
#define ALL_LEDS_OFF()      HAL_GPIO_WritePin(LED_GPIO_PORT, LED_FAULT_PIN|LED_OK_PIN, GPIO_PIN_RESET)
#define BLINK_BOTH()        do { \
    HAL_GPIO_TogglePin(LED_GPIO_PORT, LED_FAULT_PIN|LED_OK_PIN); HAL_Delay(150); \
    HAL_GPIO_TogglePin(LED_GPIO_PORT, LED_FAULT_PIN|LED_OK_PIN); HAL_Delay(150); \
} while(0)

/* ============================================================
 *  REFERENCE VECTOR  (generated from healthy.wav in Python)
 *  Layout: [MFCC_mean x13 | MFCC_std x13 | RMS | ZCR | Kurtosis]
 * ============================================================ */
static const float32_t REF_VEC[FEAT_VEC_LEN] = {
    /* --- MFCC Mean (C0 … C12) --- */
    -1625.875087f,  313.781203f,   62.972752f,  169.642213f,
      60.255603f,  108.889839f,   49.407974f,   19.941279f,
      51.763004f,   -8.176115f,   15.033925f,  -31.936357f,
      18.647681f,
    /* --- MFCC Std  (C0 … C12) --- */
     139.905050f,   47.396829f,   57.822906f,   38.670140f,
      43.153754f,   31.350061f,   23.353403f,   23.736599f,
      26.468043f,   21.769266f,   24.032569f,   19.734928f,
      17.497415f,
    /* --- Time-domain extras --- */
       0.220412f,   /* RMS      */
       0.032825f,   /* ZCR      */
       6.692044f    /* Kurtosis */
};

/* ============================================================
 *  MEL FILTERBANK  (pre-computed, stored in Flash)
 *  26 filters x 257 FFT bins  (FRAME_SIZE/2+1)
 *  Generated offline — saves ~100 ms of init time on MCU
 *
 *  NOTE: For brevity the array is initialised to zeros here.
 *        Replace with the Python-generated array from
 *        generate_filterbank.py  (provided separately).
 *        Alternatively call mel_filterbank_init() at startup
 *        which fills it at runtime — uses ~20 KB stack temp.
 * ============================================================ */
#define FFT_BINS    (FRAME_SIZE / 2 + 1)   /* 257 */
static float32_t MEL_FB[N_MELS][FFT_BINS]; /* filled by mel_filterbank_init() */

/* DCT matrix: N_MFCC x N_MELS  (13 x 26) */
static float32_t DCT_MATRIX[N_MFCC][N_MELS];

/* Hamming window */
static float32_t HAMMING[FRAME_SIZE];

/* ============================================================
 *  BUFFERS
 * ============================================================ */
/* DMA double-buffer: I2S delivers 32-bit words (24-bit left-justified) */
static int32_t  dma_buf_a[FRAME_SIZE];       /* half-complete ISR fills this */
static int32_t  dma_buf_b[FRAME_SIZE];       /* full-complete ISR fills this */

/* Full recording buffer — 80 000 float samples @ 16 kHz x 5 s */
/* Place in SRAM (STM32F411 has 128 KB) — ~320 KB needed.
   Since 80k float32 = 320 KB > 128 KB we process in STREAMING fashion:
   each DMA half/full callback feeds the DSP pipeline directly.
   We accumulate per-frame MFCC into a rolling stats buffer.          */

/* MFCC accumulator across all frames */
static float32_t mfcc_sum[N_MFCC];
static float32_t mfcc_sq_sum[N_MFCC];
static uint32_t  frame_count;

/* Time-domain accumulators */
static float32_t rms_acc;
static float32_t zcr_acc;
static float32_t kurt_acc;
static uint32_t  td_frame_count;

/* FFT instance (CMSIS-DSP) */
static arm_rfft_fast_instance_f32 fft_inst;

/* ============================================================
 *  PERIPHERAL HANDLES  (declared in main, used across callbacks)
 * ============================================================ */
I2S_HandleTypeDef  hi2s2;
DMA_HandleTypeDef  hdma_spi2_rx;

/* ============================================================
 *  STATE FLAGS
 * ============================================================ */
volatile uint8_t recording_active = 0;
volatile uint8_t half_ready       = 0;
volatile uint8_t full_ready       = 0;
volatile uint8_t recording_done   = 0;
static   uint32_t samples_captured = 0;

/* ============================================================
 *  FUNCTION PROTOTYPES
 * ============================================================ */
static void SystemClock_Config(void);
static void GPIO_Init(void);
static void I2S2_Init(void);
static void DMA_Init(void);
static void mel_filterbank_init(void);
static void dct_matrix_init(void);
static void hamming_init(void);
static void process_dma_block(int32_t *raw_buf, uint32_t len);
static void compute_frame_mfcc(float32_t *frame, float32_t *mfcc_out);
static void build_feature_vector(float32_t *feat_vec);
static float32_t cosine_similarity(const float32_t *a, const float32_t *b, uint32_t len);
static float32_t pearson_correlation(const float32_t *a, const float32_t *b, uint32_t len);
static void announce_result(float32_t cos_sim, float32_t pear_sim);
static void reset_accumulators(void);
static void blink_recording(void);

/* ============================================================
 *  MAIN
 * ============================================================ */
int main(void)
{
    HAL_Init();
    SystemClock_Config();   /* 100 MHz from HSE/PLL              */
    DMA_Init();
    GPIO_Init();
    I2S2_Init();

    /* Pre-compute lookup tables (stored in SRAM) */
    hamming_init();
    mel_filterbank_init();
    dct_matrix_init();

    /* Init CMSIS-DSP FFT for 512-point real FFT */
    arm_rfft_fast_init_f32(&fft_inst, FRAME_SIZE);

    /* Startup blink — both LEDs once */
    OK_LED_ON();    HAL_Delay(300);
    FAULT_LED_ON(); HAL_Delay(300);
    ALL_LEDS_OFF();

    while (1)
    {
        /* Wait for PA0 button press (active LOW, internal pull-up) */
        if (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0) == GPIO_PIN_RESET)
        {
            HAL_Delay(50); /* debounce */
            if (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0) == GPIO_PIN_RESET)
            {
                /* ---- Start recording session ---- */
                ALL_LEDS_OFF();
                reset_accumulators();
                samples_captured   = 0;
                recording_active   = 1;
                recording_done     = 0;

                /* Start DMA-driven I2S reception in circular double-buffer */
                HAL_I2S_Receive_DMA(&hi2s2, (uint16_t*)dma_buf_a, FRAME_SIZE * 2);

                /* Blink both LEDs while recording */
                while (!recording_done)
                {
                    blink_recording();

                    /* Process half-buffer */
                    if (half_ready)
                    {
                        half_ready = 0;
                        process_dma_block(dma_buf_a, FRAME_SIZE);
                    }
                    /* Process full-buffer */
                    if (full_ready)
                    {
                        full_ready = 0;
                        process_dma_block(dma_buf_b, FRAME_SIZE);
                    }
                    /* Stop after TOTAL_SAMPLES */
                    if (samples_captured >= TOTAL_SAMPLES)
                    {
                        HAL_I2S_DMAStop(&hi2s2);
                        recording_active = 0;
                        recording_done   = 1;
                    }
                }

                /* ---- Feature vector + similarity analysis ---- */
                float32_t feat_vec[FEAT_VEC_LEN];
                build_feature_vector(feat_vec);

                float32_t cos_sim  = cosine_similarity(feat_vec,  REF_VEC, FEAT_VEC_LEN);
                float32_t pear_sim = pearson_correlation(feat_vec, REF_VEC, FEAT_VEC_LEN);

                announce_result(cos_sim, pear_sim);

                /* Wait for button release before next capture */
                while (HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0) == GPIO_PIN_RESET)
                    HAL_Delay(10);
            }
        }
        HAL_Delay(10);
    }
}

/* ============================================================
 *  DMA CALLBACKS  (called from ISR context — keep minimal)
 * ============================================================ */

/* Half-complete: dma_buf_a has fresh data */
void HAL_I2S_RxHalfCpltCallback(I2S_HandleTypeDef *hi2s)
{
    if (!recording_active) return;
    half_ready = 1;
}

/* Full-complete: dma_buf_b has fresh data */
void HAL_I2S_RxCpltCallback(I2S_HandleTypeDef *hi2s)
{
    if (!recording_active) return;
    /* Copy second half of DMA buffer */
    memcpy(dma_buf_b, dma_buf_a + FRAME_SIZE, FRAME_SIZE * sizeof(int32_t));
    full_ready = 1;
}

/* ============================================================
 *  PROCESS ONE DMA BLOCK -> MFCC ACCUMULATION
 * ============================================================ */
static void process_dma_block(int32_t *raw_buf, uint32_t len)
{
    /* Step 1: Convert I2S 24-bit left-justified int32 -> float32 [-1, 1] */
    float32_t frame_f[FRAME_SIZE];
    float32_t sum = 0.0f;
    for (uint32_t i = 0; i < len; i++)
    {
        /* INMP441: data is in bits[31:8], shift right by 8 for 24-bit value */
        int32_t  s24  = raw_buf[i] >> 8;
        frame_f[i]    = (float32_t)s24 / 8388608.0f;  /* 2^23 */
        sum          += frame_f[i];
    }

    /* Step 2: DC offset removal */
    float32_t dc = sum / len;
    for (uint32_t i = 0; i < len; i++) frame_f[i] -= dc;

    /* Step 3: Normalize */
    float32_t max_val;
    uint32_t  max_idx;
    arm_abs_max_f32(frame_f, len, &max_val, &max_idx);
    if (max_val > 1e-6f)
    {
        float32_t inv = 1.0f / max_val;
        arm_scale_f32(frame_f, inv, frame_f, len);
    }

    /* Step 4: Time-domain features for this block */
    float32_t rms_val;
    arm_rms_f32(frame_f, len, &rms_val);
    rms_acc += rms_val;

    uint32_t zcr_count = 0;
    for (uint32_t i = 1; i < len; i++)
        if ((frame_f[i] >= 0) != (frame_f[i-1] >= 0)) zcr_count++;
    zcr_acc += (float32_t)zcr_count / len;

    /* Kurtosis accumulator */
    float32_t mean_v, var_v, kurt_v = 0.0f;
    arm_mean_f32(frame_f, len, &mean_v);
    arm_var_f32(frame_f, len, &var_v);
    if (var_v > 1e-10f)
    {
        float32_t std_v = sqrtf(var_v);
        float32_t norm_sum4 = 0.0f;
        for (uint32_t i = 0; i < len; i++)
        {
            float32_t d = (frame_f[i] - mean_v) / std_v;
            norm_sum4 += d * d * d * d;
        }
        kurt_v = norm_sum4 / len;
    }
    kurt_acc += kurt_v;
    td_frame_count++;

    /* Step 5: Apply Hamming window */
    arm_mult_f32(frame_f, HAMMING, frame_f, len);

    /* Step 6: Compute MFCC for this frame and accumulate */
    float32_t mfcc_frame[N_MFCC];
    compute_frame_mfcc(frame_f, mfcc_frame);

    for (uint32_t k = 0; k < N_MFCC; k++)
    {
        mfcc_sum[k]    += mfcc_frame[k];
        mfcc_sq_sum[k] += mfcc_frame[k] * mfcc_frame[k];
    }
    frame_count++;
    samples_captured += len;
}

/* ============================================================
 *  COMPUTE MFCC FOR ONE WINDOWED FRAME
 * ============================================================ */
static void compute_frame_mfcc(float32_t *frame, float32_t *mfcc_out)
{
    /* FFT output buffer (complex interleaved, size FRAME_SIZE) */
    float32_t fft_out[FRAME_SIZE];
    float32_t mag_sq[FFT_BINS];

    /* Real FFT (CMSIS-DSP) */
    arm_rfft_fast_f32(&fft_inst, frame, fft_out, 0);

    /* Power spectrum: |X[k]|^2 / N */
    /* fft_out[0] = DC, fft_out[1] = Nyquist (real), rest = Re,Im pairs */
    mag_sq[0] = (fft_out[0] * fft_out[0]) / FRAME_SIZE;
    mag_sq[FFT_BINS-1] = (fft_out[1] * fft_out[1]) / FRAME_SIZE;
    for (uint32_t k = 1; k < FFT_BINS - 1; k++)
    {
        float32_t re = fft_out[2*k];
        float32_t im = fft_out[2*k + 1];
        mag_sq[k] = (re*re + im*im) / FRAME_SIZE;
    }

    /* Apply Mel filterbank: mel_energy[m] = sum_k(FB[m][k] * mag_sq[k]) */
    float32_t mel_energy[N_MELS];
    for (uint32_t m = 0; m < N_MELS; m++)
    {
        arm_dot_prod_f32(MAL_FB[m], mag_sq, FFT_BINS, &mel_energy[m]);
        if (mel_energy[m] < 1e-10f) mel_energy[m] = 1e-10f;
        /* Log (dB) */
        mel_energy[m] = 20.0f * log10f(mel_energy[m]);
    }

    /* DCT: mfcc[n] = sum_m(log_mel[m] * DCT[n][m]) */
    for (uint32_t n = 0; n < N_MFCC; n++)
    {
        arm_dot_prod_f32(DCT_MATRIX[n], mel_energy, N_MELS, &mfcc_out[n]);
    }
}

/* ============================================================
 *  BUILD FEATURE VECTOR FROM ACCUMULATORS
 * ============================================================ */
static void build_feature_vector(float32_t *feat_vec)
{
    float32_t inv_fc = (frame_count > 0) ? (1.0f / frame_count) : 1.0f;
    float32_t inv_td = (td_frame_count > 0) ? (1.0f / td_frame_count) : 1.0f;

    /* MFCC mean */
    for (uint32_t k = 0; k < N_MFCC; k++)
        feat_vec[k] = mfcc_sum[k] * inv_fc;

    /* MFCC std */
    for (uint32_t k = 0; k < N_MFCC; k++)
    {
        float32_t mean_sq = feat_vec[k] * feat_vec[k];
        float32_t var = (mfcc_sq_sum[k] * inv_fc) - mean_sq;
        feat_vec[N_MFCC + k] = (var > 0.0f) ? sqrtf(var) : 0.0f;
    }

    /* Time-domain features */
    feat_vec[26] = rms_acc  * inv_td;
    feat_vec[27] = zcr_acc  * inv_td;
    feat_vec[28] = kurt_acc * inv_td;
}

/* ============================================================
 *  VECTOR SIMILARITY FUNCTIONS
 * ============================================================ */
static float32_t cosine_similarity(const float32_t *a, const float32_t *b, uint32_t len)
{
    float32_t dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (uint32_t i = 0; i < len; i++)
    {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    float32_t denom = sqrtf(norm_a) * sqrtf(norm_b);
    return (denom > 1e-10f) ? (dot / denom) : 0.0f;
}

static float32_t pearson_correlation(const float32_t *a, const float32_t *b, uint32_t len)
{
    float32_t mean_a = 0.0f, mean_b = 0.0f;
    for (uint32_t i = 0; i < len; i++) { mean_a += a[i]; mean_b += b[i]; }
    mean_a /= len; mean_b /= len;

    float32_t cov = 0.0f, var_a = 0.0f, var_b = 0.0f;
    for (uint32_t i = 0; i < len; i++)
    {
        float32_t da = a[i] - mean_a;
        float32_t db = b[i] - mean_b;
        cov   += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    float32_t denom = sqrtf(var_a) * sqrtf(var_b);
    return (denom > 1e-10f) ? (cov / denom) : 0.0f;
}

/* ============================================================
 *  RESULT ANNOUNCEMENT
 * ============================================================ */
static void announce_result(float32_t cos_sim, float32_t pear_sim)
{
    uint8_t is_fault = (cos_sim < COS_THRESHOLD) || (pear_sim < PEARSON_THRESHOLD);

    if (is_fault)
    {
        /* Blink FAULT LED 5x fast then hold ON */
        for (int i = 0; i < 5; i++)
        {
            HAL_GPIO_WritePin(LED_GPIO_PORT, LED_FAULT_PIN, GPIO_PIN_SET);
            HAL_Delay(100);
            HAL_GPIO_WritePin(LED_GPIO_PORT, LED_FAULT_PIN, GPIO_PIN_RESET);
            HAL_Delay(100);
        }
        FAULT_LED_ON();   /* PB1 stays ON */
    }
    else
    {
        /* Blink OK LED 3x slow then hold ON */
        for (int i = 0; i < 3; i++)
        {
            HAL_GPIO_WritePin(LED_GPIO_PORT, LED_OK_PIN, GPIO_PIN_SET);
            HAL_Delay(200);
            HAL_GPIO_WritePin(LED_GPIO_PORT, LED_OK_PIN, GPIO_PIN_RESET);
            HAL_Delay(200);
        }
        OK_LED_ON();      /* PB2 stays ON */
    }

    /* Optional: SWO/UART debug output */
    /* printf("COS=%.4f  PEAR=%.4f  -> %s\r\n",
              cos_sim, pear_sim, is_fault ? "FAULT" : "OK"); */
}

/* ============================================================
 *  RESET ACCUMULATORS BEFORE EACH RECORDING
 * ============================================================ */
static void reset_accumulators(void)
{
    memset(mfcc_sum,    0, sizeof(mfcc_sum));
    memset(mfcc_sq_sum, 0, sizeof(mfcc_sq_sum));
    frame_count    = 0;
    rms_acc        = 0.0f;
    zcr_acc        = 0.0f;
    kurt_acc       = 0.0f;
    td_frame_count = 0;
}

/* ============================================================
 *  RECORDING BLINK (both LEDs alternating)
 * ============================================================ */
static void blink_recording(void)
{
    static uint32_t last_tick = 0;
    static uint8_t  state     = 0;
    if (HAL_GetTick() - last_tick > 300)
    {
        last_tick = HAL_GetTick();
        state ^= 1;
        HAL_GPIO_WritePin(LED_GPIO_PORT, LED_FAULT_PIN,  state ? GPIO_PIN_SET : GPIO_PIN_RESET);
        HAL_GPIO_WritePin(LED_GPIO_PORT, LED_OK_PIN,    !state ? GPIO_PIN_SET : GPIO_PIN_RESET);
    }
}

/* ============================================================
 *  LOOKUP TABLE INIT FUNCTIONS
 * ============================================================ */
static void hamming_init(void)
{
    for (uint32_t i = 0; i < FRAME_SIZE; i++)
        HAMMING[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (FRAME_SIZE - 1));
}

static void mel_filterbank_init(void)
{
    /* Hz -> Mel */
    #define HZ2MEL(f)  (2595.0f * log10f(1.0f + (f) / 700.0f))
    #define MEL2HZ(m)  (700.0f  * (powf(10.0f, (m) / 2595.0f) - 1.0f))

    float32_t mel_lo  = HZ2MEL(0.0f);
    float32_t mel_hi  = HZ2MEL(SAMPLE_RATE / 2.0f);
    float32_t mel_pts[N_MELS + 2];
    float32_t hz_pts [N_MELS + 2];
    uint32_t  bin_pts[N_MELS + 2];

    for (uint32_t m = 0; m < N_MELS + 2; m++)
    {
        mel_pts[m] = mel_lo + (mel_hi - mel_lo) * m / (N_MELS + 1);
        hz_pts[m]  = MEL2HZ(mel_pts[m]);
        bin_pts[m] = (uint32_t)floorf((FRAME_SIZE + 1) * hz_pts[m] / SAMPLE_RATE);
        if (bin_pts[m] >= FFT_BINS) bin_pts[m] = FFT_BINS - 1;
    }

    memset(MEL_FB, 0, sizeof(MEL_FB));
    for (uint32_t m = 1; m <= N_MELS; m++)
    {
        uint32_t left   = bin_pts[m - 1];
        uint32_t center = bin_pts[m];
        uint32_t right  = bin_pts[m + 1];

        for (uint32_t k = left; k < center; k++)
            if (center != left)
                MEL_FB[m-1][k] = (float32_t)(k - left) / (center - left);

        for (uint32_t k = center; k <= right; k++)
            if (right != center)
                MEL_FB[m-1][k] = (float32_t)(right - k) / (right - center);
    }
}

static void dct_matrix_init(void)
{
    /* Type-II DCT: D[n][m] = cos(pi*n*(2m+1)/(2*N_MELS)) */
    for (uint32_t n = 0; n < N_MFCC; n++)
        for (uint32_t m = 0; m < N_MELS; m++)
            DCT_MATRIX[n][m] = cosf(M_PI * n * (2*m + 1) / (2.0f * N_MELS));
}

/* ============================================================
 *  PERIPHERAL INIT
 * ============================================================ */

static void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

    /* HSE -> PLL -> 100 MHz SYSCLK */
    RCC_OscInitStruct.OscillatorType      = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState            = RCC_HSE_ON;
    RCC_OscInitStruct.PLL.PLLState        = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource       = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM            = 25;   /* 25 MHz / 25 = 1 MHz VCO in */
    RCC_OscInitStruct.PLL.PLLN            = 200;  /* x200 = 200 MHz VCO          */
    RCC_OscInitStruct.PLL.PLLP            = RCC_PLLP_DIV2;  /* /2 = 100 MHz     */
    RCC_OscInitStruct.PLL.PLLQ            = 4;   /* USB 48 MHz (not used here)   */
    HAL_RCC_OscConfig(&RCC_OscInitStruct);

    RCC_ClkInitStruct.ClockType      = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                     | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;    /* 100 MHz AHB  */
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;      /* 50 MHz APB1  */
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;      /* 100 MHz APB2 */
    HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3);
}

static void GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    /* PA0 — User Button (input, internal pull-up) */
    GPIO_InitStruct.Pin  = GPIO_PIN_0;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* PB1 (FAULT LED) + PB2 (OK LED) — output push-pull */
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_1 | GPIO_PIN_2, GPIO_PIN_RESET);
    GPIO_InitStruct.Pin   = GPIO_PIN_1 | GPIO_PIN_2;
    GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull  = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    /* I2S2 pins: PB12=WS, PB13=CLK, PB15=SD  (AF5) */
    GPIO_InitStruct.Pin       = GPIO_PIN_12 | GPIO_PIN_13 | GPIO_PIN_15;
    GPIO_InitStruct.Mode      = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull      = GPIO_NOPULL;
    GPIO_InitStruct.Speed     = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF5_SPI2;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
}

static void DMA_Init(void)
{
    __HAL_RCC_DMA1_CLK_ENABLE();

    /* DMA1 Stream3 Channel0 — SPI2/I2S2 RX */
    hdma_spi2_rx.Instance                 = DMA1_Stream3;
    hdma_spi2_rx.Init.Channel             = DMA_CHANNEL_0;
    hdma_spi2_rx.Init.Direction           = DMA_PERIPH_TO_MEMORY;
    hdma_spi2_rx.Init.PeriphInc           = DMA_PINC_DISABLE;
    hdma_spi2_rx.Init.MemInc              = DMA_MINC_ENABLE;
    hdma_spi2_rx.Init.PeriphDataAlignment = DMA_PDATAALIGN_HALFWORD;
    hdma_spi2_rx.Init.MemDataAlignment    = DMA_MDATAALIGN_WORD;
    hdma_spi2_rx.Init.Mode                = DMA_CIRCULAR;
    hdma_spi2_rx.Init.Priority            = DMA_PRIORITY_HIGH;
    hdma_spi2_rx.Init.FIFOMode            = DMA_FIFOMODE_DISABLE;
    HAL_DMA_Init(&hdma_spi2_rx);

    __HAL_LINKDMA(&hi2s2, hdmarx, hdma_spi2_rx);

    HAL_NVIC_SetPriority(DMA1_Stream3_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(DMA1_Stream3_IRQn);
}

static void I2S2_Init(void)
{
    __HAL_RCC_SPI2_CLK_ENABLE();

    /* INMP441: I2S Philips mode, 32-bit word (24-bit data left-justified),
       Master receive, sample rate 16 kHz                                    */
    hi2s2.Instance          = SPI2;
    hi2s2.Init.Mode         = I2S_MODE_MASTER_RX;
    hi2s2.Init.Standard     = I2S_STANDARD_PHILIPS;
    hi2s2.Init.DataFormat   = I2S_DATAFORMAT_32B;
    hi2s2.Init.MCLKOutput   = I2S_MCLKOUTPUT_DISABLE;
    hi2s2.Init.AudioFreq    = I2S_AUDIOFREQ_16K;
    hi2s2.Init.CPOL         = I2S_CPOL_LOW;
    hi2s2.Init.ClockSource  = I2S_CLOCK_PLL;
    hi2s2.Init.FullDuplexMode = I2S_FULLDUPLEXMODE_DISABLE;
    HAL_I2S_Init(&hi2s2);
}

/* ============================================================
 *  IRQ HANDLERS
 * ============================================================ */
void DMA1_Stream3_IRQHandler(void)
{
    HAL_DMA_IRQHandler(&hdma_spi2_rx);
}

void SysTick_Handler(void)
{
    HAL_IncTick();
}
