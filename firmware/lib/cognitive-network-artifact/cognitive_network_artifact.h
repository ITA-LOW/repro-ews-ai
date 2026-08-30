#ifndef COGNITIVE_NETWORK_ARTIFACT_H_
#define COGNITIVE_NETWORK_ARTIFACT_H_

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi_types.h>

// Network configuration
#define MAX_NEIGHBORS 10
#define MAX_ROUTES 24
#define DISCOVERY_INTERVAL_MS 5000
#define NEIGHBOR_TIMEOUT_MS 15000
/** Period for broadcasting distance-vector (Bellman–Ford exchange). */
#define DV_INTERVAL_MS 6000
/** Metric value meaning “unreachable” in DV payloads (not 0xFFFF = reserved for link unknown). */
#define CNA_METRIC_INFINITY 0xFFFE

// Message types and capabilities
enum MessageType {
  MSG_DISCOVERY = 1,
  MSG_COOPERATION_REQUEST = 4,
  MSG_COOPERATION_RESPONSE = 5,
  MSG_FUOTA_START = 10,
  MSG_FUOTA_CHUNK = 11,
  MSG_FUOTA_END = 12,
  MSG_FUOTA_READY = 13,
  MSG_FUOTA_ACK = 14,
  MSG_FUOTA_REBOOT = 15,
  MSG_FUOTA_CHUNK_ACK = 16,
  /** Optional greeting (unicast); payload is UTF-8 text (e.g. "Hello from …"). */
  MSG_HELLO = 17,
  /**
   * Distance-vector: payload = count (1 byte) + count × (dest_mac[6], metric_le[2]).
   * Announces best known metrics to destinations (Bellman–Ford / RIP-style).
   */
  MSG_DISTANCE_VECTOR = 18
};
enum CapabilityType {
  CAP_SENSOR_LIGHT = 0x0001,
  CAP_SENSOR_ULTRASONIC = 0x0002,
  CAP_ACTUATOR_BUZZER = 0x0004,
  CAP_ACTUATOR_LED = 0x0008,
  CAP_SENSOR_TEMP_HUMID = 0x0010,  // DHT11
  CAP_SENSOR_TEMP_ANALOG = 0x0020, // LM35
  CAP_SENSOR_IR = 0x0040,          // IR Receiver
  CAP_SENSOR_POT = 0x0080,         // Potentiometer
  /** Firmware-update carrier on the mesh (extended capability, high byte). */
  CAP_FUOTA_CARRIER = 0x0100
};

// Data structures
typedef struct {
  uint8_t mac[6];
  uint32_t last_seen;
  uint16_t capabilities;
  /** Last RSSI sample (dBm) from promiscuous RX; -128 = not yet measured. */
  int8_t rssi_last;
  /**
   * EWMA of RSSI (dBm), same sentinel -128 = unknown.
   * Used with rssi_last for Bellman–Ford edge cost (Phase 2+).
   */
  int8_t rssi_ema;
  /**
   * Monotonic link cost from smoothed RSSI (lower = better path).
   * 0xFFFF while RSSI unknown.
   */
  uint16_t link_cost;
} NeighborEntry_t;

/** Best known path to a destination MAC (next hop + total metric). */
typedef struct {
  uint8_t dest[6];
  uint16_t metric;
  uint8_t next_hop[6];
} RoutingEntry_t;
typedef struct __attribute__((packed)) {
  uint8_t type;
  uint8_t sender_mac[6];
  uint8_t payload_length; // Moved before payload for alignment
  uint8_t payload[210];
} NetworkMessage_t;
typedef struct {
  uint8_t action_type;
  uint8_t param_length;
  uint8_t priority;
  uint8_t parameters[10];
} CooperationAction_t;

/** Decode capability bitmask from MSG_DISCOVERY payload (1 or 2 bytes). */
inline uint16_t parse_capabilities_from_discovery(
    const NetworkMessage_t *msg) {
  if (msg->payload_length >= 2)
    return (uint16_t)msg->payload[0] |
           ((uint16_t)msg->payload[1] << 8);
  if (msg->payload_length == 1)
    return (uint16_t)msg->payload[0];
  return 0;
}

/** Map RSSI (dBm) to a positive edge cost; weaker signal → higher cost. */
inline uint16_t cna_link_cost_from_rssi(int8_t rssi) {
  if (rssi <= -127)
    return 0xFFFF;
  int c = 100 - (int)rssi;
  if (c < 1)
    c = 1;
  if (c > 400)
    c = 400;
  return (uint16_t)c;
}

// Callback function type for data reception
typedef void (*esp_now_recv_cb_t)(const uint8_t *mac_addr, const uint8_t *data,
                                  int len);

/** How Wi-Fi is configured before ESP-NOW (coexistence with HTTP OTA on STA). */
enum class CnaWifiPreset {
  /** Standalone mesh: STA not associated, fixed channel 1 (default). */
  MeshDefault,
  /** STA already connected (e.g. WiFi.begin OK); do not disconnect or force ch1. */
  StaLinkUp
};

class CognitiveNetworkArtifact {
private:
  NeighborEntry_t neighbors[MAX_NEIGHBORS];
  uint8_t neighbor_count;

  RoutingEntry_t routes[MAX_ROUTES];
  uint8_t route_count;
  unsigned long last_dv_sent_ms_;

  uint8_t agent_mac[6];
  uint16_t agent_capabilities;
  char agent_name[16];
  bool is_initialized;

  esp_now_recv_cb_t user_recv_cb_;

  static void on_data_sent_static(const uint8_t *mac_addr,
                                  esp_now_send_status_t status);

  void init_neighbor_metrics_(NeighborEntry_t *n);
  void update_neighbor_rssi_(const uint8_t *mac, int8_t rssi);
  void install_promiscuous_rssi_tap_();

  void init_routing_self_();
  void sync_direct_routes_from_neighbors_();
  void purge_routes_for_lost_neighbor_(const uint8_t *mac);
  int find_route_index_(const uint8_t *dest) const;
  bool upsert_route_(const uint8_t *dest, uint16_t metric, const uint8_t *next_hop);
  void process_distance_vector_(const uint8_t *from_mac, const uint8_t *payload,
                                uint8_t payload_len);
  bool send_distance_vector_broadcast_();

  static CognitiveNetworkArtifact *s_instance_;
  static void promisc_rx_cb_(void *buf, wifi_promiscuous_pkt_type_t type);
  static void recv_bridge_cb_(const uint8_t *mac_addr, const uint8_t *data,
                              int len);

public:
  CognitiveNetworkArtifact();

  bool initialize(const char *agent_name, uint16_t capabilities,
                  esp_now_recv_cb_t callback,
                  CnaWifiPreset preset = CnaWifiPreset::MeshDefault);
  void start();

  const uint8_t *get_agent_mac();
  void add_or_update_neighbor(const uint8_t *mac, uint16_t capabilities);
  void touch_neighbor(const uint8_t *mac);
  void cleanup_neighbors();

  bool send_discovery_broadcast();
  bool send_cooperation_request(const uint8_t *target_mac,
                                CooperationAction_t *action);
  bool send_cooperation_response(const uint8_t *target_mac, uint8_t request_id,
                                 bool success);

  /** Unicast a short text hello to a peer (must be in neighbor table / ESP-NOW peers). */
  bool send_hello(const uint8_t *target_mac, const char *text);

  bool find_neighbor_by_capability(uint16_t capability,
                                   NeighborEntry_t *neighbor);
  void print_neighbor_table();

  /** Call each loop iteration: sends DV broadcast when DV_INTERVAL_MS elapsed. */
  void send_distance_vector_if_due();

  /** Debug: routes learned via neighbors + DV (next hop + metric). */
  void print_routing_table();
};

#endif
