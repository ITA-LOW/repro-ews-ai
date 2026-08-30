#include "cognitive_network_artifact.h"
#include <cstring>
#include <esp_err.h>
#include <esp_log.h>
#include <esp_wifi.h>

static const char *TAG = "CognitiveNetworkArtifact";

CognitiveNetworkArtifact *CognitiveNetworkArtifact::s_instance_ = nullptr;

// Static variable for broadcast
static uint8_t broadcast_mac[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

CognitiveNetworkArtifact::CognitiveNetworkArtifact() {
  neighbor_count = 0;
  route_count = 0;
  last_dv_sent_ms_ = 0;
  user_recv_cb_ = nullptr;
  is_initialized = false;
}

bool CognitiveNetworkArtifact::initialize(const char *agent_name,
                                          uint16_t capabilities,
                                          esp_now_recv_cb_t callback,
                                          CnaWifiPreset preset) {
  s_instance_ = this;
  user_recv_cb_ = callback;

  strncpy(this->agent_name, agent_name, sizeof(this->agent_name) - 1);
  this->agent_name[sizeof(this->agent_name) - 1] = '\0';
  this->agent_capabilities = capabilities;

  if (preset == CnaWifiPreset::MeshDefault) {
    /* Weak USB / long cable: 5 V rail can dip when WiFi/RF starts → brownout reset.
     * Pause + lower TX power reduces peak current. If you never see [CNA] Agent: …,
     * the board reset during WiFi.mode — use a shorter cable or powered hub. */
    Serial.printf("[CNA] Radio bring-up for \"%s\"...\n", this->agent_name);
    Serial.flush();
    delay(500);
    WiFi.mode(WIFI_STA);
    delay(300);
#if defined(WIFI_POWER_8_5dBm)
    WiFi.setTxPower(WIFI_POWER_8_5dBm);
#elif defined(WIFI_POWER_5dBm)
    WiFi.setTxPower(WIFI_POWER_5dBm);
#endif
    esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
    WiFi.disconnect();
    delay(50);
  } else {
    Serial.printf(
        "[CNA] ESP-NOW with existing Wi-Fi STA (coexistence) for \"%s\"...\n",
        this->agent_name);
    Serial.flush();
    delay(200);
  }

  esp_read_mac(agent_mac, ESP_MAC_WIFI_STA);
  Serial.printf(
      "[CNA] Agent: %s, MAC: %02X:%02X:%02X:%02X:%02X:%02X, Channel: 1\n",
      this->agent_name, agent_mac[0], agent_mac[1], agent_mac[2], agent_mac[3],
      agent_mac[4], agent_mac[5]);

  init_routing_self_();

  delay(100);
  if (esp_now_init() != ESP_OK) {
    Serial.println("[CNA] ERROR: Failed to initialize ESP-NOW");
    return false;
  }

  esp_now_register_send_cb(on_data_sent_static);
  esp_now_register_recv_cb(recv_bridge_cb_);

  // Add broadcast peer
  esp_now_peer_info_t peer_info = {};
  memcpy(peer_info.peer_addr, broadcast_mac, 6);
  if (!esp_now_is_peer_exist(broadcast_mac)) {
    if (esp_now_add_peer(&peer_info) != ESP_OK) {
      Serial.println("[CNA] ERROR: Failed to add broadcast peer");
      return false;
    }
  }

  install_promiscuous_rssi_tap_();

  Serial.println("[CNA] Distance-vector (Bellman–Ford) enabled; call "
                 "send_distance_vector_if_due() from loop.");

  is_initialized = true;
  return true;
}

void CognitiveNetworkArtifact::init_routing_self_() {
  route_count = 0;
  memcpy(routes[0].dest, agent_mac, 6);
  routes[0].metric = 0;
  memcpy(routes[0].next_hop, agent_mac, 6);
  route_count = 1;
}

void CognitiveNetworkArtifact::sync_direct_routes_from_neighbors_() {
  for (int i = 0; i < neighbor_count; i++) {
    if (neighbors[i].link_cost >= 0xFFFF)
      continue;
    upsert_route_(neighbors[i].mac, neighbors[i].link_cost, neighbors[i].mac);
  }
}

int CognitiveNetworkArtifact::find_route_index_(const uint8_t *dest) const {
  for (int i = 0; i < route_count; i++) {
    if (memcmp(routes[i].dest, dest, 6) == 0)
      return i;
  }
  return -1;
}

bool CognitiveNetworkArtifact::upsert_route_(const uint8_t *dest, uint16_t metric,
                                            const uint8_t *next_hop) {
  if (metric >= CNA_METRIC_INFINITY)
    return false;

  int idx = find_route_index_(dest);
  if (idx >= 0) {
    if (routes[idx].metric <= metric)
      return false;
    routes[idx].metric = metric;
    memcpy(routes[idx].next_hop, next_hop, 6);
    return true;
  }
  if (route_count >= MAX_ROUTES) {
    int worst = -1;
    uint16_t worst_m = 0;
    for (int j = 0; j < route_count; j++) {
      if (memcmp(routes[j].dest, agent_mac, 6) == 0)
        continue;
      if (worst < 0 || routes[j].metric > worst_m) {
        worst = j;
        worst_m = routes[j].metric;
      }
    }
    if (worst < 0)
      return false;
    memcpy(routes[worst].dest, dest, 6);
    routes[worst].metric = metric;
    memcpy(routes[worst].next_hop, next_hop, 6);
    return true;
  }
  memcpy(routes[route_count].dest, dest, 6);
  routes[route_count].metric = metric;
  memcpy(routes[route_count].next_hop, next_hop, 6);
  route_count++;
  return true;
}

void CognitiveNetworkArtifact::purge_routes_for_lost_neighbor_(
    const uint8_t *mac) {
  int w = 0;
  for (int i = 0; i < route_count; i++) {
    if (memcmp(routes[i].dest, agent_mac, 6) == 0) {
      routes[w++] = routes[i];
      continue;
    }
    if (memcmp(routes[i].dest, mac, 6) == 0 ||
        memcmp(routes[i].next_hop, mac, 6) == 0)
      continue;
    routes[w++] = routes[i];
  }
  route_count = (uint8_t)w;
}

void CognitiveNetworkArtifact::process_distance_vector_(const uint8_t *from_mac,
                                                        const uint8_t *payload,
                                                        uint8_t payload_len) {
  if (!payload || payload_len < 1)
    return;

  uint16_t edge = 0xFFFF;
  for (int i = 0; i < neighbor_count; i++) {
    if (memcmp(neighbors[i].mac, from_mac, 6) == 0) {
      edge = neighbors[i].link_cost;
      break;
    }
  }
  if (edge >= 0xFFFF)
    return;

  uint8_t n = payload[0];
  if (payload_len < (size_t)(1 + n * 8))
    return;

  for (uint8_t i = 0; i < n; i++) {
    const uint8_t *e = payload + 1 + i * 8;
    uint16_t adv = (uint16_t)e[6] | ((uint16_t)e[7] << 8);
    if (adv >= CNA_METRIC_INFINITY)
      continue;
    if (memcmp(e, agent_mac, 6) == 0)
      continue;
    uint32_t total = (uint32_t)edge + (uint32_t)adv;
    if (total >= (uint32_t)CNA_METRIC_INFINITY)
      continue;
    upsert_route_(e, (uint16_t)total, from_mac);
  }
}

bool CognitiveNetworkArtifact::send_distance_vector_broadcast_() {
  sync_direct_routes_from_neighbors_();

  uint8_t pl[210];
  uint8_t n_entries = 0;
  size_t off = 1;
  for (int i = 0; i < route_count && n_entries < 25; i++) {
    if (off + 8 > sizeof(pl))
      break;
    memcpy(pl + off, routes[i].dest, 6);
    pl[off + 6] = (uint8_t)(routes[i].metric & 0xFF);
    pl[off + 7] = (uint8_t)((routes[i].metric >> 8) & 0xFF);
    off += 8;
    n_entries++;
  }
  pl[0] = n_entries;

  NetworkMessage_t msg = {};
  msg.type = MSG_DISTANCE_VECTOR;
  memcpy(msg.sender_mac, agent_mac, 6);
  msg.payload_length = (uint8_t)off;
  memcpy(msg.payload, pl, off);

  size_t send_len = sizeof(uint8_t) + 6 + 1 + off;
  return esp_now_send(broadcast_mac, (uint8_t *)&msg, send_len) == ESP_OK;
}

void CognitiveNetworkArtifact::send_distance_vector_if_due() {
  unsigned long now = millis();
  if (now - last_dv_sent_ms_ < DV_INTERVAL_MS)
    return;
  last_dv_sent_ms_ = now;
  if (send_distance_vector_broadcast_()) {
    /* optional: one-line debug
    Serial.printf("[CNA] DV broadcast (%u routes)\n", (unsigned)route_count);
    */
  }
}

void CognitiveNetworkArtifact::recv_bridge_cb_(const uint8_t *mac_addr,
                                               const uint8_t *data, int len) {
  if (!s_instance_ || !data || len < 8) {
    if (s_instance_ && s_instance_->user_recv_cb_)
      s_instance_->user_recv_cb_(mac_addr, data, len);
    return;
  }

  uint8_t payload_len = data[7];
  if (len < 8 + payload_len) {
    if (s_instance_->user_recv_cb_)
      s_instance_->user_recv_cb_(mac_addr, data, len);
    return;
  }

  uint8_t type = data[0];
  uint8_t sender_mac[6];
  memcpy(sender_mac, data + 1, 6);

  if (memcmp(sender_mac, s_instance_->agent_mac, 6) == 0)
    return;

  if (type == MSG_DISTANCE_VECTOR) {
    s_instance_->process_distance_vector_(sender_mac, data + 8, payload_len);
    s_instance_->touch_neighbor(sender_mac);
    return;
  }

  if (s_instance_->user_recv_cb_)
    s_instance_->user_recv_cb_(mac_addr, data, len);
}

void CognitiveNetworkArtifact::print_routing_table() {
  unsigned shown = 0;
  for (int i = 0; i < route_count; i++) {
    if (memcmp(routes[i].dest, agent_mac, 6) == 0)
      continue;
    shown++;
  }
  Serial.printf("--- Routing Table (%u shown; self-route kept for DV) ---\n",
                shown);
  int line = 0;
  for (int i = 0; i < route_count; i++) {
    if (memcmp(routes[i].dest, agent_mac, 6) == 0)
      continue;
    Serial.printf(
        "Route %d: dest %02X:%02X:%02X:%02X:%02X:%02X | metric %u | via "
        "%02X:%02X:%02X:%02X:%02X:%02X\n",
        line++, routes[i].dest[0], routes[i].dest[1], routes[i].dest[2],
        routes[i].dest[3], routes[i].dest[4], routes[i].dest[5],
        (unsigned)routes[i].metric, routes[i].next_hop[0], routes[i].next_hop[1],
        routes[i].next_hop[2], routes[i].next_hop[3], routes[i].next_hop[4],
        routes[i].next_hop[5]);
  }
  Serial.println("-----------------------------");
}

void CognitiveNetworkArtifact::init_neighbor_metrics_(NeighborEntry_t *n) {
  n->rssi_last = -128;
  n->rssi_ema = -128;
  n->link_cost = 0xFFFF;
}

void CognitiveNetworkArtifact::update_neighbor_rssi_(const uint8_t *mac,
                                                     int8_t rssi) {
  if (!mac)
    return;
  for (int i = 0; i < neighbor_count; i++) {
    if (memcmp(neighbors[i].mac, mac, 6) != 0)
      continue;
    neighbors[i].rssi_last = rssi;
    if (neighbors[i].rssi_ema <= -127)
      neighbors[i].rssi_ema = rssi;
    else
      neighbors[i].rssi_ema = (int8_t)(((int)neighbors[i].rssi_ema * 7 + (int)rssi) / 8);
    neighbors[i].link_cost = cna_link_cost_from_rssi(neighbors[i].rssi_ema);
    neighbors[i].last_seen = millis();
    sync_direct_routes_from_neighbors_();
    return;
  }
}

void CognitiveNetworkArtifact::promisc_rx_cb_(void *buf,
                                              wifi_promiscuous_pkt_type_t type) {
  if (!s_instance_ || type != WIFI_PKT_MGMT || !buf)
    return;
  auto *pkt = static_cast<wifi_promiscuous_pkt_t *>(buf);
  if (pkt->rx_ctrl.sig_len < 24)
    return;
  int8_t rssi = static_cast<int8_t>(pkt->rx_ctrl.rssi);
  const uint8_t *frame = pkt->payload;
  const uint8_t *ta = frame + 10;
  if ((ta[0] & 0x01) != 0)
    return;
  s_instance_->update_neighbor_rssi_(ta, rssi);
}

void CognitiveNetworkArtifact::install_promiscuous_rssi_tap_() {
  s_instance_ = this;
  wifi_promiscuous_filter_t filt = {};
  filt.filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT;
  esp_err_t e = esp_wifi_set_promiscuous_filter(&filt);
  if (e != ESP_OK) {
    Serial.printf("[CNA] esp_wifi_set_promiscuous_filter failed: %s\n",
                  esp_err_to_name(e));
    return;
  }
  e = esp_wifi_set_promiscuous_rx_cb(promisc_rx_cb_);
  if (e != ESP_OK) {
    Serial.printf("[CNA] esp_wifi_set_promiscuous_rx_cb failed: %s\n",
                  esp_err_to_name(e));
    return;
  }
  e = esp_wifi_set_promiscuous(true);
  if (e != ESP_OK) {
    Serial.printf("[CNA] esp_wifi_set_promiscuous(true) failed: %s\n",
                  esp_err_to_name(e));
    return;
  }
  Serial.println(
      "[CNA] RSSI: promiscuous tap on (mgmt only) — edge cost from neighbor EMA.");
}

void CognitiveNetworkArtifact::start() {
  Serial.println("[CNA] Manager started.");
}

void CognitiveNetworkArtifact::on_data_sent_static(
    const uint8_t *mac_addr, esp_now_send_status_t status) {
  if (status != ESP_NOW_SEND_SUCCESS) {
    ESP_LOGW(TAG, "Send failure to %02X:%02X:%02X:%02X:%02X:%02X", mac_addr[0],
             mac_addr[1], mac_addr[2], mac_addr[3], mac_addr[4], mac_addr[5]);
  }
}

const uint8_t *CognitiveNetworkArtifact::get_agent_mac() { return agent_mac; }

bool CognitiveNetworkArtifact::send_discovery_broadcast() {
  NetworkMessage_t msg;
  msg.type = MSG_DISCOVERY;
  memcpy(msg.sender_mac, agent_mac, 6);
  msg.payload[0] = (uint8_t)(agent_capabilities & 0xFF);
  msg.payload[1] = (uint8_t)((agent_capabilities >> 8) & 0xFF);
  msg.payload_length = 2;

  return esp_now_send(broadcast_mac, (uint8_t *)&msg,
                      sizeof(uint8_t) + 6 + msg.payload_length + 1) == ESP_OK;
}

bool CognitiveNetworkArtifact::send_cooperation_request(
    const uint8_t *target_mac, CooperationAction_t *action) {
  NetworkMessage_t msg;
  msg.type = MSG_COOPERATION_REQUEST;
  memcpy(msg.sender_mac, agent_mac, 6);
  memcpy(msg.payload, action, sizeof(CooperationAction_t));
  msg.payload_length = sizeof(CooperationAction_t);

  return esp_now_send(target_mac, (uint8_t *)&msg, sizeof(uint8_t) + 6 + msg.payload_length + 1) == ESP_OK;
}

bool CognitiveNetworkArtifact::send_cooperation_response(
    const uint8_t *target_mac, uint8_t request_id, bool success) {
  NetworkMessage_t msg;
  msg.type = MSG_COOPERATION_RESPONSE;
  memcpy(msg.sender_mac, agent_mac, 6);
  msg.payload[0] = request_id;
  msg.payload[1] = (uint8_t)success;
  msg.payload_length = 2;

  return esp_now_send(target_mac, (uint8_t *)&msg, sizeof(uint8_t) + 6 + msg.payload_length + 1) == ESP_OK;
}

bool CognitiveNetworkArtifact::send_hello(const uint8_t *target_mac,
                                          const char *text) {
  if (!text)
    return false;
  NetworkMessage_t msg = {};
  msg.type = MSG_HELLO;
  memcpy(msg.sender_mac, agent_mac, 6);
  size_t n = strlen(text);
  if (n > sizeof(msg.payload))
    n = sizeof(msg.payload);
  memcpy(msg.payload, text, n);
  msg.payload_length = (uint8_t)n;
  return esp_now_send(target_mac, (uint8_t *)&msg,
                      sizeof(uint8_t) + 6 + 1 + msg.payload_length) == ESP_OK;
}

void CognitiveNetworkArtifact::add_or_update_neighbor(const uint8_t *mac,
                                                      uint16_t capabilities) {
  for (int i = 0; i < neighbor_count; i++) {
    if (memcmp(neighbors[i].mac, mac, 6) == 0) {
      neighbors[i].last_seen = millis();
      neighbors[i].capabilities = capabilities;
      return;
    }
  }

  if (neighbor_count < MAX_NEIGHBORS) {
    esp_now_peer_info_t peer_info = {};
    memcpy(peer_info.peer_addr, mac, 6);
    peer_info.channel = 0;

    if (!esp_now_is_peer_exist(mac)) {
      esp_err_t err = esp_now_add_peer(&peer_info);
      if (err == ESP_OK) {
        memcpy(neighbors[neighbor_count].mac, mac, 6);
        neighbors[neighbor_count].capabilities = capabilities;
        init_neighbor_metrics_(&neighbors[neighbor_count]);
        neighbors[neighbor_count].last_seen = millis();
        neighbor_count++;
        Serial.printf(
            "[CNA] New neighbor added: %02X:%02X:%02X:%02X:%02X:%02X\n", mac[0],
            mac[1], mac[2], mac[3], mac[4], mac[5]);
      } else {
        Serial.printf("[CNA] esp_now_add_peer failed: %s\n", esp_err_to_name(err));
      }
    }
  }
}

void CognitiveNetworkArtifact::touch_neighbor(const uint8_t *mac) {
  for (int i = 0; i < neighbor_count; i++) {
    if (memcmp(neighbors[i].mac, mac, 6) == 0) {
      neighbors[i].last_seen = millis();
      return;
    }
  }
}

void CognitiveNetworkArtifact::cleanup_neighbors() {
  unsigned long now = millis();
  for (int i = neighbor_count - 1; i >= 0; i--) {
    if ((now - neighbors[i].last_seen) > NEIGHBOR_TIMEOUT_MS) {
      Serial.printf(
          "[CNA] Removing inactive neighbor: %02X:%02X:%02X:%02X:%02X:%02X\n",
          neighbors[i].mac[0], neighbors[i].mac[1], neighbors[i].mac[2],
          neighbors[i].mac[3], neighbors[i].mac[4], neighbors[i].mac[5]);
      uint8_t gone[6];
      memcpy(gone, neighbors[i].mac, 6);
      esp_now_del_peer(neighbors[i].mac);
      purge_routes_for_lost_neighbor_(gone);
      if (i != neighbor_count - 1) {
        memcpy(&neighbors[i], &neighbors[neighbor_count - 1],
               sizeof(NeighborEntry_t));
      }
      neighbor_count--;
    }
  }
}

bool CognitiveNetworkArtifact::find_neighbor_by_capability(
    uint16_t capability, NeighborEntry_t *neighbor) {
  if (!neighbor)
    return false;
  for (int i = 0; i < neighbor_count; i++) {
    if ((neighbors[i].capabilities & capability) != 0) {
      memcpy(neighbor, &neighbors[i], sizeof(NeighborEntry_t));
      return true;
    }
  }
  return false;
}

void CognitiveNetworkArtifact::print_neighbor_table() {
  Serial.printf("--- Neighbor Table (%d) ---\n", neighbor_count);
  for (int i = 0; i < neighbor_count; i++) {
    if (neighbors[i].rssi_ema <= -127) {
      Serial.printf(
          "Neighbor %d: %02X:%02X:%02X:%02X:%02X:%02X | Caps: 0x%04X | RSSI: "
          "(pending) | cost: —\n",
          i, neighbors[i].mac[0], neighbors[i].mac[1], neighbors[i].mac[2],
          neighbors[i].mac[3], neighbors[i].mac[4], neighbors[i].mac[5],
          neighbors[i].capabilities);
    } else {
      Serial.printf(
          "Neighbor %d: %02X:%02X:%02X:%02X:%02X:%02X | Caps: 0x%04X | "
          "RSSI: %d dBm (ema %d) | cost: %u\n",
          i, neighbors[i].mac[0], neighbors[i].mac[1], neighbors[i].mac[2],
          neighbors[i].mac[3], neighbors[i].mac[4], neighbors[i].mac[5],
          neighbors[i].capabilities, (int)neighbors[i].rssi_last,
          (int)neighbors[i].rssi_ema, (unsigned)neighbors[i].link_cost);
    }
  }
  Serial.println("-----------------------------");
}
