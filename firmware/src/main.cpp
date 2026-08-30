#include "cart_model.h"
#include "cognitive_network_artifact.h"
#include "poisson_belief.h"
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>

// --- INTEGRAÇÃO COM MOTOR BDI (embedded-bdi) ---
#include "agent.h"
#include "belief_base.h"
#include "event_base.h"
#include "functions.h"
#include "intention_base.h"
#include "plan_base.h"

// --- SUPORTE AO DISPLAY OLED HELTEC V3 ---
#include "SSD1306Wire.h"

// Pinos do Display OLED Heltec V3
#define Vext_PIN 36
#define RST_OLED 21
#define SDA_OLED 17
#define SCL_OLED 18

static SSD1306Wire display(0x3c, SDA_OLED, SCL_OLED, GEOMETRY_128_64, I2C_ONE);

// ==============================================================================
// CONFIGURAÇÕES DE HARDWARE (Wokwi / Heltec ESP32-S3 LoRa V3)
// ==============================================================================
#define NODE_ID 0x0001
#define PIN_ADC_VOLTAGE 4 // Tensao Solar (V) - GPIO4 (ADC1_CH3)
#define PIN_ADC_CURRENT 5 // Corrente Solar (A) - GPIO5 (ADC1_CH4)
#define PIN_PWM_BOOST 6   // Saida PWM do MPPT - GPIO6 (PWM Out)

// Pinos limpos e livres de acordo com o Datasheet Oficial Heltec V3
#define PIN_BATTERY 7  // Tensao da Bateria (0-100%) - GPIO7 (ADC1_CH6)
#define PIN_TEMP 8     // Temperatura DHT22 - GPIO19 (Digital I/O)
#define PIN_HUMIDITY 9 // Umidade DHT22 - Compartilhado no GPIO19
#define PIN_PRECIP 10  // Chuva Pluviometro - GPIO3 (Interrupcao Digital)
#define PIN_SOIL 11    // Umidade do Solo - GPIO2 (ADC1_CH1)

#define PWM_FREQ 31250
#define PWM_RES 8 // 8-bit (0-255)
#define PWM_CHANNEL 0

// ==============================================================================
// ESTRUTURA DO PACOTE DE ALERTA (EwsAlertPacket)
// ==============================================================================
enum DisasterType : uint8_t {
  DISASTER_NONE = 0,
  DISASTER_LANDSLIDE = 1, // Risco Crítico de Deslizamento de Terra
  DISASTER_FLOOD = 2,     // Risco Crítico de Alagamento / Inundação
  DISASTER_MULTIPLE = 3   // Risco Múltiplo (Deslizamento + Alagamento)
};

struct __attribute__((packed)) EwsAlertPacket {
  uint16_t node_id;       // ID único do nó sensor (ex: 0x0001)
  uint8_t disaster_type;  // Tipo de Desastre (Deslizamento / Alagamento)
  uint8_t action_code;    // 1=ESP-NOW, 2=LoRa WARNING, 5=WATCH start, 6=WATCH end
  float risk_probability; // Probabilidade calculada pela IA (0.0 a 1.0)
  float battery_soc;      // SoC % da Bateria
  float precip_mm;        // Precipitação em mm
  float soil_moisture;    // Umidade do Solo %
  uint32_t uptime_ms;     // Ticks de tempo no ESP32
};

// ==============================================================================
// PROPOSIÇÕES E CRENÇAS FORMAIS DO AGENTE BDI
// ==============================================================================
#define PROP_TINYML_HAZARD 0            // +perception_hazard (CART class != 0)
#define PROP_BATTERY_LOW 1              // +battery_low(SoC < 20%)
#define PROP_NEIGHBOR_RELAY_AVAILABLE 2 // +neighbor_relay_available (ESP-NOW)
#define PROP_LORA_GATEWAY_REACHABLE                                            \
  3 // +lora_gateway_reachable [PLACEHOLDER LORA MESH]
#define PROP_SATELLITE_UPLINK_READY                                            \
  4                        // +satellite_uplink_ready [PLACEHOLDER SATÉLITE]
#define PROP_DOWNLINK_RX_EXPECTED 5 // +downlink_rx_expected (DART)
#define PROP_POISSON_WATCH 6        // +poisson_watch (P_risk crossed 0.20 up)
#define PROP_START_GOAL 10 // +!start (Objetivo Inicial)

// --- CRENÇAS DE LONGO PRAZO PERSISTIDAS NA RTC SLOW SRAM (DEEP SLEEP SURVIVAL)
// ---
RTC_DATA_ATTR uint32_t rtc_boot_count = 0;
RTC_DATA_ATTR float rtc_battery_soc = 70.0f;
RTC_DATA_ATTR float rtc_cumulative_hazard = 0.0f;
RTC_DATA_ATTR uint32_t rtc_deep_sleep_interval_sec =
    3600; // Intervalo configurável: 600s (10m), 1800s (30m), 3600s (1h)

// Variáveis Globais de Estado para Construção Dinâmica de Pacotes no BDI
EwsAlertPacket g_current_alert_packet;

// Belief-state trackers used by edge generation and plan actions.
static bool bdi_state_hazard = false;
static bool bdi_state_poisson_watch = false;
static bool bdi_state_bat_low = false;
static bool bdi_state_neighbor_relay = false;
static bool bdi_state_downlink_expected = false;

// Artefato Cognitivo de Rede do mbed-mas (Distance-Vector Bellman-Ford + RSSI
// EWMA)
CognitiveNetworkArtifact cna;

// Declarações antecipadas de funções de transmissão e display
void setup_oled();
void update_oled_screen(float risk_pct, float bat_pct, const char *action_str,
                       bool hazard, bool watch);
void send_espnow_alert(const EwsAlertPacket &pkt,
                       const uint8_t *target_mac = nullptr);
void send_lora_alert(const EwsAlertPacket &pkt);

// ==============================================================================
// AÇÕES DO AGENTE BDI (BDI Action Handlers)
// ==============================================================================
bool action_show_ready_message() {
  Serial.println("[BDI ACTION] Agente EWS-AI inicializado! Motor BDI pronto.");
  return true;
}

bool action_deep_sleep() {
  rtc_deep_sleep_interval_sec = bdi_state_poisson_watch ? 300 : 3600;
  rtc_boot_count++;
  Serial.printf("[BDI ACTION] (Impacto 0.5mW) Risco Normal (Boot #%u). "
                "Hibernando em Deep Sleep por %u s...\n",
                rtc_boot_count, rtc_deep_sleep_interval_sec);
  Serial.flush();
  Serial.println("[MODO BANCADA] Deep Sleep desativado temporariamente para "
                 "monitoramento continuo no display OLED.");
  // esp_sleep_enable_timer_wakeup((uint64_t)rtc_deep_sleep_interval_sec *
  // 1000000ULL); esp_deep_sleep_start();
  return true;
}

bool action_espnow_relay() {
  rtc_deep_sleep_interval_sec = 120;
  Serial.println("[BDI ACTION] (Impacto 45mW) ALERTA! Bateria Baixa. "
                 "Transmitindo via ESP-NOW Peer Relay Mesh.");
  g_current_alert_packet.action_code = 1;
  send_espnow_alert(g_current_alert_packet);
  return true;
}

bool action_lora_direct_alert() {
  rtc_deep_sleep_interval_sec = 120;
  Serial.println("[BDI ACTION] (Impacto 360mW) ALERTA CRÍTICO! Disparando "
                 "rádio LoRa (915MHz / SX1262).");
  g_current_alert_packet.action_code = 2;
  send_lora_alert(g_current_alert_packet);
  return true;
}

bool action_suspend_tx_wait_dart() {
  Serial.println("[BDI ACTION] (Impacto 15mW) Downlink Satelite DART Previsto! "
                 "Suspendendo Uplink LoRa para evitar colisao de Pseudo-Ortogonalidade.");
  g_current_alert_packet.action_code = 4;
  return true;
}

/* Rising/falling edge only. Placeholder radio (Serial). Not a FLoRaSat result. */
bool action_start_watch() {
  if (!bdi_state_hazard) {
    rtc_deep_sleep_interval_sec = 300;
  }
  Serial.printf("[BDI ACTION] WATCH START  P_risk=%.3f  "
                "(decision-maker: agent started watching)\n",
                g_current_alert_packet.risk_probability);
  g_current_alert_packet.action_code = 5;
  send_lora_alert(g_current_alert_packet);
  return true;
}

bool action_end_watch() {
  if (!bdi_state_hazard) {
    rtc_deep_sleep_interval_sec = 3600;
  }
  Serial.printf("[BDI ACTION] WATCH END  P_risk=%.3f  "
                "(decision-maker: watch cleared)\n",
                g_current_alert_packet.risk_probability);
  g_current_alert_packet.action_code = 6;
  send_lora_alert(g_current_alert_packet);
  return true;
}

// ==============================================================================
// CONFIGURAÇÃO DA CLASSE DE CONFIGURAÇÃO DO AGENTE BDI (EwsAgentSettings)
// ==============================================================================
class EwsAgentSettings {
private:
  Body body_0, body_1, body_2, body_3, body_4, body_5, body_6, body_7;
  Context context_0, context_1, context_2, context_3, context_4, context_5,
      context_6, context_7;
  BeliefBase belief_base;
  EventBase event_base;
  PlanBase plan_base;
  IntentionBase intention_base;

public:
  EwsAgentSettings() {
    belief_base = BeliefBase(16);
    event_base = EventBase(10);
    plan_base = PlanBase(8);
    intention_base = IntentionBase(12, 5);

    // --- Adicionar Crenças Iniciais ---
    belief_base.add_belief(
        Belief(PROP_TINYML_HAZARD, nullptr, false)); // tinyml_hazard
    belief_base.add_belief(
        Belief(PROP_BATTERY_LOW, nullptr, false)); // battery_low
    belief_base.add_belief(Belief(PROP_NEIGHBOR_RELAY_AVAILABLE, nullptr,
                                  false)); // neighbor_relay
    belief_base.add_belief(Belief(PROP_LORA_GATEWAY_REACHABLE, nullptr,
                                  true)); // lora_reachable (Default)
    belief_base.add_belief(Belief(PROP_SATELLITE_UPLINK_READY, nullptr,
                                  false)); // satellite_ready (Placeholder)
    belief_base.add_belief(Belief(PROP_DOWNLINK_RX_EXPECTED, nullptr,
                                  false)); // downlink_rx_expected (DART)
    belief_base.add_belief(
        Belief(PROP_POISSON_WATCH, nullptr, false)); // watch (not a warning)
    belief_base.add_belief(
        Belief(PROP_START_GOAL, nullptr, true)); // start goal

    // --- Adicionar Evento Inicial ---
    event_base.add_event(Event(EventOperator::GOAL_ADDITION, PROP_START_GOAL));

    // --- PLANOS BDI (AgentSpeak) ---

    // Plano 0: +!start <- action_show_ready_message.
    Proposition prop_0(PROP_START_GOAL);
    context_0 = Context(0);
    body_0 = Body(1);
    body_0.add_instruction(BodyInstruction(BodyType::ACTION, Proposition(100),
                                           action_show_ready_message));
    plan_base.add_plan(
        Plan(EventOperator::GOAL_ADDITION, prop_0, &context_0, &body_0));

    // Plano 1: -tinyml_hazard <- action_deep_sleep.
    // Trigger: Risco Normal (< 0.20)
    Proposition prop_1(PROP_TINYML_HAZARD);
    context_1 = Context(0);
    body_1 = Body(1);
    body_1.add_instruction(
        BodyInstruction(BodyType::ACTION, Proposition(101), action_deep_sleep));
    plan_base.add_plan(
        Plan(EventOperator::BELIEF_DELETION, prop_1, &context_1, &body_1));

    // Plano 2: +tinyml_hazard : battery_low & neighbor_relay_available <-
    // action_espnow_relay. Trigger: Risco Alto + Bateria Fraca + Vizinho Mesh
    // Presente
    Proposition prop_2(PROP_TINYML_HAZARD);
    context_2 = Context(2);
    context_2.add_context(
        ContextCondition(Proposition(PROP_BATTERY_LOW), false));
    context_2.add_context(
        ContextCondition(Proposition(PROP_NEIGHBOR_RELAY_AVAILABLE), false));
    body_2 = Body(1);
    body_2.add_instruction(BodyInstruction(BodyType::ACTION, Proposition(102),
                                           action_espnow_relay));
    plan_base.add_plan(
        Plan(EventOperator::BELIEF_ADDITION, prop_2, &context_2, &body_2));

    // Plano 3: +tinyml_hazard : not battery_low & not downlink_rx_expected <- action_lora_direct_alert.
    // Trigger: Risco Alto + Bateria Normal + Sem conflito DART
    Proposition prop_3(PROP_TINYML_HAZARD);
    context_3 = Context(2);
    context_3.add_context(ContextCondition(Proposition(PROP_BATTERY_LOW),
                                           true)); // not battery_low
    context_3.add_context(ContextCondition(Proposition(PROP_DOWNLINK_RX_EXPECTED),
                                           true)); // not downlink_rx_expected
    body_3 = Body(1);
    body_3.add_instruction(BodyInstruction(BodyType::ACTION, Proposition(103),
                                           action_lora_direct_alert));
    plan_base.add_plan(
        Plan(EventOperator::BELIEF_ADDITION, prop_3, &context_3, &body_3));

    // Plano 4: +downlink_rx_expected : true <- action_suspend_tx_wait_dart.
    // Trigger: Passagem de satélite identificada pelo FLoRaSat (DART constraint)
    Proposition prop_4(PROP_DOWNLINK_RX_EXPECTED);
    context_4 = Context(0); // true (sem condição extra)
    body_4 = Body(1);
    body_4.add_instruction(BodyInstruction(BodyType::ACTION, Proposition(104),
                                           action_suspend_tx_wait_dart));
    plan_base.add_plan(
        Plan(EventOperator::BELIEF_ADDITION, prop_4, &context_4, &body_4));

    // Plano 5: +poisson_watch <- action_start_watch.
    // Rising edge of P_risk >= 0.20. WATCH packet, not a CART warning.
    Proposition prop_5(PROP_POISSON_WATCH);
    context_5 = Context(0);
    body_5 = Body(1);
    body_5.add_instruction(BodyInstruction(BodyType::ACTION, Proposition(105),
                                           action_start_watch));
    plan_base.add_plan(
        Plan(EventOperator::BELIEF_ADDITION, prop_5, &context_5, &body_5));

    // Plano 6: -poisson_watch <- action_end_watch.
    Proposition prop_6(PROP_POISSON_WATCH);
    context_6 = Context(0);
    body_6 = Body(1);
    body_6.add_instruction(
        BodyInstruction(BodyType::ACTION, Proposition(106), action_end_watch));
    plan_base.add_plan(
        Plan(EventOperator::BELIEF_DELETION, prop_6, &context_6, &body_6));

    // Plano 7: +tinyml_hazard : battery_low & not neighbor & not downlink <-
    // action_lora_direct_alert. Fallback when peer relay is unavailable.
    Proposition prop_7(PROP_TINYML_HAZARD);
    context_7 = Context(3);
    context_7.add_context(
        ContextCondition(Proposition(PROP_BATTERY_LOW), false));
    context_7.add_context(ContextCondition(
        Proposition(PROP_NEIGHBOR_RELAY_AVAILABLE), true));
    context_7.add_context(ContextCondition(
        Proposition(PROP_DOWNLINK_RX_EXPECTED), true));
    body_7 = Body(1);
    body_7.add_instruction(BodyInstruction(
        BodyType::ACTION, Proposition(103), action_lora_direct_alert));
    plan_base.add_plan(
        Plan(EventOperator::BELIEF_ADDITION, prop_7, &context_7, &body_7));
  }

  BeliefBase *get_belief_base() { return &belief_base; }
  EventBase *get_event_base() { return &event_base; }
  PlanBase *get_plan_base() { return &plan_base; }
  IntentionBase *get_intention_base() { return &intention_base; }
};

// Instâncias Globais do Agente BDI
EwsAgentSettings agent_settings;
Agent bdi_agent(agent_settings.get_belief_base(),
                agent_settings.get_event_base(), agent_settings.get_plan_base(),
                agent_settings.get_intention_base());

// Traces gerados pelo Simulador OMNeT++ FLoRaSat
// Representa janelas de contato do satélite (em segundos de uptime)
const uint32_t florasat_contact_windows[][2] = {
    {3600, 3610},   // Satélite passa na hora 1
    {18000, 18015}, // Satélite passa na hora 5
    {43200, 43212}  // Satélite passa na hora 12
};
const int florasat_num_windows = 3;

// ==============================================================================
// PERCEPTION ARTIFACT N — unscaled CART (physical units, static table)
// ==============================================================================
RTC_DATA_ATTR float rtc_precip_ring[24] = {0};
RTC_DATA_ATTR uint8_t rtc_precip_count = 0;
RTC_DATA_ATTR uint8_t rtc_precip_head = 0;

static void precip_push(float p1h) {
  rtc_precip_ring[rtc_precip_head] = p1h;
  rtc_precip_head = (uint8_t)((rtc_precip_head + 1) % 24);
  if (rtc_precip_count < 24) {
    rtc_precip_count++;
  }
}

static float precip_sum_hours(int hours) {
  int n = rtc_precip_count;
  if (n <= 0) {
    return 0.0f;
  }
  if (hours > n) {
    hours = n;
  }
  float acc = 0.0f;
  for (int k = 1; k <= hours; k++) {
    int idx = (int)rtc_precip_head - k;
    if (idx < 0) {
      idx += 24;
    }
    acc += rtc_precip_ring[idx];
  }
  return acc;
}

void setup_perception() {
  Serial.printf("[Perception] CART %d nodes, %d features, static table.\n",
                CART_N_NODES, CART_N_FEATURES);
}

int run_perception(const float x[CART_N_FEATURES]) { return cart_predict(x); }

// ==============================================================================
// PILAR 2: PSO-MPPT COM RESET DINÂMICO (DMR)
// ==============================================================================
#define NUM_PARTICULAS 20
#define MAXD 0.975
#define MIND 0.05
#define CicloAvaliacao 125
#define CicloReset 250

typedef struct {
  float posicao;
  float velocidade;
  float melhorPos;
  float melhorPot;
  float fitness;
} Particle;

Particle particulas[NUM_PARTICULAS];
float W = 0.2, C1 = 0.6, C2 = 0.6;
float D = 0.5;
float melhorPos = 0, melhorPot = 0;
int particulaAtual = 0, avaliacaoParticula = 0;
bool pso_inicializado = false;

float margem_dmr = 0.02f;
int count_ciclo_reset = 0;
float media_pot_reset = 0.0f;
int count_reset = 0;
int count_flag = 0;
bool dmr_flag_estabilizado = false;

float ler_potencia_painel() {
  float v = analogRead(PIN_ADC_VOLTAGE) * (3.3 / 4095.0) * 10.0;
  float i = analogRead(PIN_ADC_CURRENT) * (3.3 / 4095.0);
  return v * i;
}

void inicializar_pso() {
  float multi = ((MAXD - MIND) / NUM_PARTICULAS);
  for (int i = 0; i < NUM_PARTICULAS; i++) {
    particulas[i].posicao = MAXD - (i * multi);
    particulas[i].velocidade = 0.0;
    particulas[i].melhorPos = MAXD - (i * multi);
    particulas[i].melhorPot = 0.0;
    particulas[i].fitness = 0.0;
  }
  melhorPot = 0.0f;
  melhorPos = 0.5f;
  particulaAtual = 0;
  avaliacaoParticula = 0;
  dmr_flag_estabilizado = false;
  count_reset = 0;
  count_flag = 0;
  pso_inicializado = true;
}

void verificar_dmr(float potencia_atual) {
  if (dmr_flag_estabilizado &&
      (potencia_atual > melhorPot * (1.0f + margem_dmr) ||
       potencia_atual < melhorPot * (1.0f - margem_dmr))) {
    Serial.println(
        "[MPPT-DMR] Oscilação solar detectada! Resetando enxame PSO...");
    inicializar_pso();
    return;
  }

  count_ciclo_reset++;
  media_pot_reset += potencia_atual;

  if (count_ciclo_reset >= CicloReset) {
    media_pot_reset /= CicloReset;

    if (media_pot_reset <= melhorPot * (1.0f + margem_dmr) &&
        media_pot_reset >= melhorPot * (1.0f - margem_dmr)) {
      count_reset = 0;
      count_flag++;
      if (count_flag >= 5) {
        dmr_flag_estabilizado = true;
      }
    } else {
      count_reset++;
      count_flag = 0;
      if (count_reset >= 3 && dmr_flag_estabilizado) {
        Serial.println(
            "[MPPT-DMR] Perda de GMPP! Reiniciando busca fotovoltaica.");
        inicializar_pso();
      }
    }
    count_ciclo_reset = 0;
    media_pot_reset = 0.0f;
  }
}

void rodar_pso_step(float potencia_atual) {
  if (!pso_inicializado)
    inicializar_pso();
  verificar_dmr(potencia_atual);

  avaliacaoParticula++;
  if (avaliacaoParticula >= CicloAvaliacao) {
    particulas[particulaAtual].fitness /= CicloAvaliacao;
    avaliacaoParticula = 0;

    if (particulas[particulaAtual].fitness > melhorPot) {
      melhorPot = particulas[particulaAtual].fitness;
      melhorPos = particulas[particulaAtual].posicao;
    }

    if (particulas[particulaAtual].fitness >
        particulas[particulaAtual].melhorPot) {
      particulas[particulaAtual].melhorPot = particulas[particulaAtual].fitness;
      particulas[particulaAtual].melhorPos = particulas[particulaAtual].posicao;
    }

    float r = (float)random(100) / 100.0f;
    particulas[particulaAtual].velocidade =
        (W * particulas[particulaAtual].velocidade) +
        (C1 * r *
         (particulas[particulaAtual].melhorPos -
          particulas[particulaAtual].posicao)) +
        (C2 * 1.0f * (melhorPos - particulas[particulaAtual].posicao));

    particulas[particulaAtual].posicao += particulas[particulaAtual].velocidade;

    particulaAtual++;
    if (particulaAtual >= NUM_PARTICULAS)
      particulaAtual = 0;
  } else {
    particulas[particulaAtual].fitness += potencia_atual;
  }

  D = particulas[particulaAtual].posicao;
  if (D > MAXD)
    D = MAXD;
  if (D < MIND)
    D = MIND;

  ledcWrite(PWM_CHANNEL, (uint32_t)(D * 255.0f));
}

// ==============================================================================
// PILAR 3: COMUNICAÇÃO ESP-NOW & PLACEHOLDERS LORA / SATÉLITE
// ==============================================================================
uint8_t broadcast_mac[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

void OnDataRecv(const uint8_t *mac_addr, const uint8_t *incomingData, int len) {
  if (len == sizeof(EwsAlertPacket)) {
    EwsAlertPacket pkt;
    memcpy(&pkt, incomingData, sizeof(pkt));

    if (pkt.node_id == NODE_ID ||
        (memcmp(mac_addr, cna.get_agent_mac(), 6) == 0)) {
      return; // Ignorar loopback local
    }

    Serial.printf("[ESP-NOW RX Relay] Pacote de Emergência Recebido do Nó "
                  "0x%04X! Bateria: %.1f%%\n",
                  pkt.node_id, pkt.battery_soc);
    Serial.println("[ESP-NOW RX Relay] Assumindo como Relay Ativo! Repassando "
                   "mensagem via rádio LoRa...");

    send_lora_alert(pkt);
  }
}

void setup_espnow() {
  uint16_t capabilities =
      CAP_ACTUATOR_BUZZER | CAP_SENSOR_TEMP_HUMID | CAP_FUOTA_CARRIER;
  bool ok = cna.initialize("EWS_AI_Node", capabilities, OnDataRecv);
  if (ok) {
    cna.start();
    Serial.println("[CNA mbed-mas] Artefato Cognitivo de Rede (Bellman-Ford / "
                   "RSSI EWMA) Inicializado!");
  } else {
    Serial.println("[CNA Erro] Falha ao inicializar o mbed-mas CNA.");
  }
}

bool register_unicast_peer(const uint8_t *peer_mac) {
  if (esp_now_is_peer_exist(peer_mac))
    return true;

  esp_now_peer_info_t peer_info = {};
  memcpy(peer_info.peer_addr, peer_mac, 6);
  peer_info.channel = 0;
  peer_info.encrypt = false;
  return (esp_now_add_peer(&peer_info) == ESP_OK);
}

void send_espnow_alert(const EwsAlertPacket &pkt, const uint8_t *target_mac) {
  const uint8_t *dest_mac =
      (target_mac != nullptr) ? target_mac : broadcast_mac;

  if (target_mac != nullptr) {
    register_unicast_peer(target_mac);
    Serial.printf("[ESP-NOW TX Unicast] Enviando pedido de Relay para Vizinho "
                  "MAC: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  target_mac[0], target_mac[1], target_mac[2], target_mac[3],
                  target_mac[4], target_mac[5]);
  } else {
    Serial.println("[ESP-NOW TX Broadcast] Transmitindo Alerta de Emergência "
                   "na rede Mesh...");
  }

  esp_err_t result = esp_now_send(dest_mac, (const uint8_t *)&pkt, sizeof(pkt));
  if (result == ESP_OK) {
    Serial.printf(
        "[ESP-NOW TX OK] Pacote transmitido com sucesso (%u bytes).\n",
        sizeof(pkt));
  } else {
    Serial.printf("[ESP-NOW Erro] Falha ao transmitir pacote. Código: %d\n",
                  result);
  }
}

void setup_lora() {
  Serial.println(
      "[LoRa SX1262 - PLACEHOLDER] Interface LoRa configurada (915 MHz).");
}

void send_lora_alert(const EwsAlertPacket &pkt) {
  const char *kind = "TX";
  if (pkt.action_code == 5) {
    kind = "WATCH START";
  } else if (pkt.action_code == 6) {
    kind = "WATCH END";
  } else if (pkt.action_code == 2) {
    kind = "WARNING";
  }
  Serial.printf("[LORA PLACEHOLDER] %s | NodeID: 0x%04X | P_risk: %.1f%% | "
                "disaster: %u | action: %u\n",
                kind, pkt.node_id, pkt.risk_probability * 100.0f,
                pkt.disaster_type, pkt.action_code);
}

// ==============================================================================
// TAREFAS FREERTOS & LOOP DE RACIOCÍNIO BDI
// ==============================================================================
void task_MPPT(void *pvParameters) {
  static int print_counter = 0;
  while (1) {
    float power = ler_potencia_painel();
    rodar_pso_step(power);

    print_counter++;
    if (print_counter >= 50) {
      Serial.printf(
          "[MPPT] Potencia: %.2f W | Duty (D): %.2f | Estabilizado DMR: %s\n",
          power, D, dmr_flag_estabilizado ? "SIM" : "NAO");
      print_counter = 0;
    }
    vTaskDelay(20 / portTICK_PERIOD_MS);
  }
}

void task_AI_BDI(void *pvParameters) {
  static unsigned long last_discovery_tx = 0;
  static unsigned long last_neighbor_cleanup = 0;

  while (1) {
    unsigned long now = millis();

    // 1. Manutenção Periódica do Artefato Cognitivo de Rede (CNA)
    cna.send_distance_vector_if_due();

    if (now - last_discovery_tx > DISCOVERY_INTERVAL_MS) {
      last_discovery_tx = now;
      cna.send_discovery_broadcast();
    }

    if (now - last_neighbor_cleanup > NEIGHBOR_TIMEOUT_MS) {
      last_neighbor_cleanup = now;
      cna.cleanup_neighbors(); // Expurga vizinhos caídos e limpa rotas do
                               // Bellman-Ford
    }

    // 2. Sensors in physical units (CART is unscaled).
    float temp_c = map(analogRead(PIN_TEMP), 0, 4095, -10, 50);
    float hum_pct = map(analogRead(PIN_HUMIDITY), 0, 4095, 0, 100);
    float prec_mm = map(analogRead(PIN_PRECIP), 0, 4095, 0, 300);
    float soil_pct = map(analogRead(PIN_SOIL), 0, 4095, 0, 100);
    float bat_pct = map(analogRead(PIN_BATTERY), 0, 4095, 0, 100);
    float solar_mw = ler_potencia_painel() * 1000.0f;

    // Field: one rain/λ sample per wake. OLED bench: every cycle so Wokwi sliders move λ.
#if LORA_DISPLAY_OLED
    precip_push(prec_mm);
#else
    static bool sampled_this_wake = false;
    if (!sampled_this_wake) {
      precip_push(prec_mm);
      sampled_this_wake = true;
    }
#endif
    float precip_6h = precip_sum_hours(6);
    float precip_24h = precip_sum_hours(24);
    float lam = poisson_lambda(precip_6h, precip_24h, soil_pct);
#if LORA_DISPLAY_OLED
    poisson_push_lambda(lam);
#else
    static bool lambda_this_wake = false;
    if (!lambda_this_wake) {
      poisson_push_lambda(lam);
      lambda_this_wake = true;
    }
#endif
    float p_risk = poisson_p_risk();
    bool current_watch = (p_risk >= POI_WATCH_THR);

    float x[CART_N_FEATURES] = {temp_c, hum_pct, precip_6h, precip_24h,
                                soil_pct};
    int cart_class = run_perception(x);
    bool current_hazard = cart_is_hazard(cart_class);

    DisasterType disaster = DISASTER_NONE;
    if (cart_class == 1) {
      disaster = DISASTER_FLOOD;
    } else if (cart_class == 2) {
      disaster = DISASTER_LANDSLIDE;
    }

    // 5. Construção do Pacote de Alerta Global
    g_current_alert_packet.node_id = NODE_ID;
    g_current_alert_packet.disaster_type = static_cast<uint8_t>(disaster);
    g_current_alert_packet.risk_probability = p_risk;
    g_current_alert_packet.battery_soc = bat_pct;
    g_current_alert_packet.precip_mm = prec_mm;
    g_current_alert_packet.soil_moisture = soil_pct;
    g_current_alert_packet.uptime_ms = millis();

    // ==============================================================================
    // SINCRONIZAÇÃO DE CRENÇAS E NAVEGAÇÃO NA REDE MESH (Bellman-Ford / CNA)
    // ==============================================================================

    // 1. Perception hazard belief (CART class != 0) — WARNING
    if (current_hazard != bdi_state_hazard) {
      bdi_state_hazard = current_hazard;
      if (current_hazard) {
        agent_settings.get_event_base()->add_event(Event(
            EventOperator::BELIEF_ADDITION, Proposition(PROP_TINYML_HAZARD)));
      } else {
        agent_settings.get_event_base()->add_event(Event(
            EventOperator::BELIEF_DELETION, Proposition(PROP_TINYML_HAZARD)));
      }
    }

    // 1b. Poisson watch belief (P_risk >= 0.20) — WATCH, not a named warning
    if (current_watch != bdi_state_poisson_watch) {
      bdi_state_poisson_watch = current_watch;
      if (current_watch) {
        agent_settings.get_event_base()->add_event(Event(
            EventOperator::BELIEF_ADDITION, Proposition(PROP_POISSON_WATCH)));
      } else {
        agent_settings.get_event_base()->add_event(Event(
            EventOperator::BELIEF_DELETION, Proposition(PROP_POISSON_WATCH)));
      }
    }

    // 2. Atualizar Crença de Bateria Baixa (+battery_low)
    bool current_bat_low = (bat_pct < 20.0f);
    if (current_bat_low != bdi_state_bat_low) {
      bdi_state_bat_low = current_bat_low;
      if (current_bat_low) {
        agent_settings.get_event_base()->add_event(Event(
            EventOperator::BELIEF_ADDITION, Proposition(PROP_BATTERY_LOW)));
      } else {
        agent_settings.get_event_base()->add_event(Event(
            EventOperator::BELIEF_DELETION, Proposition(PROP_BATTERY_LOW)));
      }
    }

    // 3. Atualizar Crença de Vizinho de Relay Disponível
    // (+neighbor_relay_available com filtro RSSI EWMA)
    NeighborEntry_t relay_neighbor;
    bool neighbor_found =
        cna.find_neighbor_by_capability(CAP_ACTUATOR_BUZZER, &relay_neighbor);
    if (neighbor_found != bdi_state_neighbor_relay) {
      bdi_state_neighbor_relay = neighbor_found;
      if (neighbor_found) {
        agent_settings.get_event_base()->add_event(
            Event(EventOperator::BELIEF_ADDITION,
                  Proposition(PROP_NEIGHBOR_RELAY_AVAILABLE)));
      } else {
        agent_settings.get_event_base()->add_event(
            Event(EventOperator::BELIEF_DELETION,
                  Proposition(PROP_NEIGHBOR_RELAY_AVAILABLE)));
      }
    }

    // 4. Checagem dos Traces FLoRaSat (DART Timing)
    uint32_t uptime_sec = millis() / 1000;
    bool in_contact_window = false;
    for (int i = 0; i < florasat_num_windows; i++) {
      if (uptime_sec >= florasat_contact_windows[i][0] && uptime_sec <= florasat_contact_windows[i][1]) {
        in_contact_window = true;
        break;
      }
    }
    
    if (in_contact_window != bdi_state_downlink_expected) {
      bdi_state_downlink_expected = in_contact_window;
      if (in_contact_window) {
        agent_settings.get_event_base()->add_event(
            Event(EventOperator::BELIEF_ADDITION,
                  Proposition(PROP_DOWNLINK_RX_EXPECTED)));
      } else {
        agent_settings.get_event_base()->add_event(
            Event(EventOperator::BELIEF_DELETION,
                  Proposition(PROP_DOWNLINK_RX_EXPECTED)));
      }
    }

    // ==============================================================================
    // EXECUÇÃO DO CICLO DE RACIOCÍNIO DO AGENTE BDI (Agent::run)
    // ==============================================================================
    Serial.printf("\n[Sensores] Temp: %.1fC | Hum: %.1f%% | Chuva: %.1fmm | "
                  "Solo: %.1f%% | Bat: %.1f%%\n",
                  temp_c, hum_pct, prec_mm, soil_pct, bat_pct);
    Serial.printf("[CART] class=%d hazard=%d disaster=%u  P6=%.1f P24=%.1f\n",
                  cart_class, current_hazard ? 1 : 0, disaster, precip_6h,
                  precip_24h);
    Serial.printf("[POISSON] lambda=%.4f  P_risk=%.3f  watch=%d  duty=%us\n",
                  lam, p_risk, current_watch ? 1 : 0,
                  rtc_deep_sleep_interval_sec);

    const char *action_name;
    if (current_hazard) {
      action_name = (bat_pct < 20.0f) ? "WARNING Relay" : "WARNING LoRa";
    } else if (current_watch) {
      action_name = "WATCH";
    } else {
      action_name = "Idle";
    }
    update_oled_screen(p_risk * 100.0f, bat_pct, action_name,
                       current_hazard, current_watch);

    Serial.print(">> EXECUTANDO CICLO DELIBERATIVO DO AGENTE BDI: ");
    bdi_agent.run(); // Executa o ciclo de raciocínio estático BDI
    Serial.println("---------------------------------------------------------");

    vTaskDelay(3000 / portTICK_PERIOD_MS);
  }
}

void setup_oled() {
  pinMode(Vext_PIN, OUTPUT);
  digitalWrite(Vext_PIN, LOW); // Liga alimentação Vext da tela OLED Heltec V3
  delay(100);

  pinMode(RST_OLED, OUTPUT);
  digitalWrite(RST_OLED, LOW);
  delay(20);
  digitalWrite(RST_OLED, HIGH);
  delay(20);

  display.init();
  display.flipScreenVertically();
  display.setFont(ArialMT_Plain_10);
  display.clear();
  display.drawString(0, 0, "=== EWS-AI NODE ===");
  display.drawString(0, 16, "BDI Engine: PRONTO");
  display.drawString(0, 32, "CART perception: OK");
  display.drawString(0, 48, "Heltec V3 Inicializado");
  display.display();
}

void update_oled_screen(float risk_pct, float bat_pct, const char *action_str,
                       bool hazard, bool watch) {
  display.clear();
  display.setFont(ArialMT_Plain_10);
  display.drawString(0, 0, "--- EWS-AI NODE #0001 ---");

  const char *mode = hazard ? "[ALERT]" : (watch ? "[WATCH]" : "[OK]");
  char buf1[32];
  snprintf(buf1, sizeof(buf1), "P_risk : %.1f%% %s", risk_pct, mode);
  display.drawString(0, 16, buf1);

  char buf2[32];
  snprintf(buf2, sizeof(buf2), "Bateria  : %.0f%% (LiFePO4)", bat_pct);
  display.drawString(0, 32, buf2);

  char buf3[32];
  snprintf(buf3, sizeof(buf3), "Acao BDI : %s", action_str);
  display.drawString(0, 48, buf3);

  display.display();
}

// ==============================================================================
// SETUP E LOOP PRINCIPAL
// ==============================================================================
void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("Iniciando EWS-AI Bare-Metal Edge Node (BDI + CART + CNA)...");

  setup_oled();

  ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RES);
  ledcAttachPin(PIN_PWM_BOOST, PWM_CHANNEL);

  setup_perception();
  setup_espnow();
  setup_lora();

  xTaskCreatePinnedToCore(task_MPPT, "MPPT_Task", 4096, NULL, 2, NULL, 0);
  xTaskCreatePinnedToCore(task_AI_BDI, "AI_BDI_Task", 8192, NULL, 1, NULL, 1);
}

void loop() { vTaskDelete(NULL); }
