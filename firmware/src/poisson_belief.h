/* SPDX-License-Identifier: MIT */
/* Same λ(t) as configs/experiment.yaml / src/ews_ai/dataset.py. No date bonus. */
#pragma once

#include <math.h>
#include <stdint.h>

#define POI_LAMBDA0 0.005f
#define POI_ALPHA 1.2f
#define POI_BETA 1.5f
#define POI_GAMMA 1.8f
#define POI_P_CRIT_6H 35.0f
#define POI_P_CRIT_24H 60.0f
#define POI_SM_SAT 65.0f
#define POI_EXP_CLIP 10.0f
#define POI_HORIZON 6
#define POI_WATCH_THR 0.20f

#ifndef RTC_DATA_ATTR
#define RTC_DATA_ATTR
#endif

RTC_DATA_ATTR static float rtc_lambda_ring[POI_HORIZON] = {0};
RTC_DATA_ATTR static uint8_t rtc_lambda_count = 0;
RTC_DATA_ATTR static uint8_t rtc_lambda_head = 0;

static inline float poisson_lambda(float precip_6h, float precip_24h, float soil_pct) {
  float exponent = POI_ALPHA * (precip_6h / POI_P_CRIT_6H) +
                   POI_BETA * (precip_24h / POI_P_CRIT_24H) +
                   POI_GAMMA * (soil_pct / POI_SM_SAT);
  if (exponent > POI_EXP_CLIP) {
    exponent = POI_EXP_CLIP;
  }
  float lam = POI_LAMBDA0 * (expf(exponent) - 1.0f);
  return lam > 0.0f ? lam : 0.0f;
}

static inline void poisson_push_lambda(float lam) {
  rtc_lambda_ring[rtc_lambda_head] = lam;
  rtc_lambda_head = (uint8_t)((rtc_lambda_head + 1) % POI_HORIZON);
  if (rtc_lambda_count < POI_HORIZON) {
    rtc_lambda_count++;
  }
}

static inline float poisson_p_risk(void) {
  float acc = 0.0f;
  for (uint8_t i = 0; i < rtc_lambda_count; i++) {
    acc += rtc_lambda_ring[i];
  }
  float p = 1.0f - expf(-acc);
  if (p < 0.0f) {
    return 0.0f;
  }
  if (p > 1.0f) {
    return 1.0f;
  }
  return p;
}
