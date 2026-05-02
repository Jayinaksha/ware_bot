#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/int32.h>
#include <std_msgs/msg/float32.h>
#include <geometry_msgs/msg/twist_stamped.h>
#include <geometry_msgs/msg/vector3.h>
#include <Arduino.h>
#include <math.h>

// ==========================
// 1. PIN CONFIG
// ==========================

// MD10C Motor Pins
#define R_DIR D12  
#define R_PWM D10 
#define L_DIR D8  
#define L_PWM D6  

// Encoder Pins
#define L_ENCA A0
#define L_ENCB A2
#define R_ENCA A4
#define R_ENCB D2

// --- STEPPER PINS (From your snippet) ---
#define STEP_PIN D4
#define DIR_PIN  D14
#define EN_PIN   A3

// Robot Constants
double rod_length = 0.4;
#define TICKS_PER_REV 600
#define WHEEL_DIAMETER 0.15
#define PI 3.14159265359
const double METERS_PER_TICK = (PI * WHEEL_DIAMETER) / TICKS_PER_REV;
const double VEL_MAX = 2.74;   

// ==========================
// 2. GLOBALS
// ==========================
rcl_publisher_t debug2;
geometry_msgs__msg__Vector3 debug2_msg;
rcl_publisher_t debug_right;
geometry_msgs__msg__Vector3 debug_right_msg;

rcl_subscription_t cmd_sub;
geometry_msgs__msg__TwistStamped cmd_msg;

// --- STEPPER SUBSCRIBER ---
rcl_subscription_t sub_stepper;
std_msgs__msg__Int32 msg_stepper;

rclc_executor_t executor;
rcl_allocator_t allocator;
rclc_support_t support; 
rcl_node_t node;

// DC Motor Globals
volatile double target_vel_left = 0.0;
volatile double target_vel_right = 0.0;
volatile long left_encoder_count = 0;
volatile long right_encoder_count = 0;
volatile int last_L_A = 0;
volatile int last_R_A = 0;
long prev_left_count = 0;
long prev_right_count = 0;
unsigned long last_pid_time = 0;
const int PID_INTERVAL_MS = 50;

// --- STEPPER LOGIC GLOBALS ---
volatile bool stepper_running = false;
unsigned long last_step_time = 0;
const int step_delay = 25; // Logic from your snippet (speed)

// ==========================
// 3. PID CLASS
// ==========================
class SimplePID {
private:
  float kp, ki, kd, integral, prev_error;
public:
  SimplePID(float p, float i, float d) : kp(p), ki(i), kd(d), integral(0), prev_error(0) {}
  void reset() { integral = 0; prev_error = 0; }
  float compute(float target, float measured) {
    float error = target - measured;
    float P = kp * error;
    integral += error;
    if (integral > 0.5) integral = 0.5;
    if (integral < -0.5) integral = -0.5;
    float I = ki * integral;
    float D = kd * (error - prev_error);
    prev_error = error;
    return P + I + D;
  }
};

SimplePID pidLeft(0.3, 0.00, 0.5);
SimplePID pidRight(0.25, 0.03, 0.35);

// ==========================
// 4. INTERRUPTS & HELPER
// ==========================
void readEncoderLeft() {
  int A = digitalRead(L_ENCA);
  int B = digitalRead(L_ENCB);
  if (A != last_L_A) { (A == B) ? left_encoder_count++ : left_encoder_count--; }
  last_L_A = A;
}

void readEncoderRight() {
  int A = digitalRead(R_ENCA);
  int B = digitalRead(R_ENCB);
  if (A != last_R_A) { (A == B) ? right_encoder_count++ : right_encoder_count--; }
  last_R_A = A;
}

static inline void drive_md10c(int dir_pin, int pwm_pin, int pwm_val) {
  if (pwm_val > 0) {
    digitalWrite(dir_pin, HIGH);
    analogWrite(pwm_pin, pwm_val);
  } else if (pwm_val < 0) {
    digitalWrite(dir_pin, LOW);
    analogWrite(pwm_pin, -pwm_val);
  } else {
    analogWrite(pwm_pin, 0); 
  }
}

static inline double clamp(double v, double low, double high) {
  if (v < low) return low;
  if (v > high) return high;
  return v;
}

void cmd_callback(const void * msgin) {
  const geometry_msgs__msg__TwistStamped * msg = (const geometry_msgs__msg__TwistStamped *) msgin;
  double lx = msg->twist.linear.x;
  double az = msg->twist.angular.z;
  target_vel_left = lx - (rod_length/2.0 * az);
  target_vel_right = lx + (rod_length/2.0 * az);
}

// --- MODIFIED STEPPER CALLBACK ---
void stepper_callback(const void * msgin) {
  const std_msgs__msg__Int32 * msg = (const std_msgs__msg__Int32 *) msgin;
  int32_t command = msg->data;

  // Logic: 800 -> UP(CW), -800 -> DOWN(CCW), 0 -> STOP
  if (command == 800) {
    digitalWrite(DIR_PIN, HIGH); // CW (Up)
    stepper_running = true;
  } 
  else if (command == -800) {
    digitalWrite(DIR_PIN, LOW);  // CCW (Down)
    stepper_running = true;
  } 
  else {
    // Stop for 0 or any other number
    stepper_running = false;
  }
}

// ==========================
// 6. SETUP
// ==========================
void setup() {
  // DC Motor Pins
  pinMode(R_DIR, OUTPUT); pinMode(L_DIR, OUTPUT); 
  pinMode(R_PWM, OUTPUT); pinMode(L_PWM, OUTPUT);
  pinMode(L_ENCA, INPUT_PULLUP); pinMode(L_ENCB, INPUT_PULLUP);
  pinMode(R_ENCA, INPUT_PULLUP); pinMode(R_ENCB, INPUT_PULLUP);
  
  // --- STEPPER SETUP ---
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);
  
  digitalWrite(EN_PIN, LOW); // Enable driver
  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);

  // Interrupts
  last_L_A = digitalRead(L_ENCA); last_R_A = digitalRead(R_ENCA);
  attachInterrupt(digitalPinToInterrupt(L_ENCA), readEncoderLeft, CHANGE);
  attachInterrupt(digitalPinToInterrupt(R_ENCA), readEncoderRight, CHANGE);

  pinMode(LED_BUILTIN, OUTPUT);
  set_microros_transports(); 
  
  // Wait for Agent
  while (RMW_RET_OK != rmw_uros_ping_agent(100, 1)) {
      digitalWrite(LED_BUILTIN, HIGH); delay(250);
      digitalWrite(LED_BUILTIN, LOW); delay(250);
  }
  
  allocator = rcl_get_default_allocator();
  rclc_support_init(&support, 0, NULL, &allocator);
  rclc_node_init_default(&node, "micro_ros_stm32_md10c", "", &support);
  
  // Init Messages
  geometry_msgs__msg__Vector3__init(&debug2_msg);       
  geometry_msgs__msg__Vector3__init(&debug_right_msg);  
  geometry_msgs__msg__TwistStamped__init(&cmd_msg);
  std_msgs__msg__Int32__init(&msg_stepper);

  // Init Publishers/Subscribers
  rclc_publisher_init_default(&debug2, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Vector3), "debug2");
  rclc_publisher_init_default(&debug_right, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Vector3), "debug_right"); 
  rclc_subscription_init_default(&cmd_sub, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, TwistStamped), "cmd_vel");
  
  // Stepper Subscriber
  rclc_subscription_init_default(&sub_stepper, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32), "cmd_stepper");
  
  // Executor
  rclc_executor_init(&executor, &support.context, 2, &allocator);
  rclc_executor_add_subscription(&executor, &cmd_sub, &cmd_msg, &cmd_callback, ON_NEW_DATA);
  rclc_executor_add_subscription(&executor, &sub_stepper, &msg_stepper, &stepper_callback, ON_NEW_DATA);
}

// ==========================
// 7. LOOP
// ==========================
void loop() {
  int MIN_PWM = 17; 

  // Handle Communication
  rclc_executor_spin_some(&executor, RCL_MS_TO_NS(1));

  // --- STEPPER MOTOR LOGIC ---
  if (stepper_running) {
    unsigned long now = micros();
    if (now - last_step_time >= step_delay) {
      last_step_time = now;
      digitalWrite(STEP_PIN, HIGH);
      delayMicroseconds(5); // Small pulse width
      digitalWrite(STEP_PIN, LOW);
    }
  }

  // --- DC MOTOR PID LOGIC ---
  if (millis() - last_pid_time >= PID_INTERVAL_MS) {
    double dt = (millis() - last_pid_time) / 1000.0;
    last_pid_time = millis();

    long curr_L = left_encoder_count;
    long curr_R = right_encoder_count;
    double current_vel_left = -((curr_L - prev_left_count) * METERS_PER_TICK) / dt;
    double current_vel_right = ((curr_R - prev_right_count) * METERS_PER_TICK) / dt;
    prev_left_count = curr_L; prev_right_count = curr_R;

    int pwm_left = 0; int pwm_right = 0;

    // LEFT PID
    if (fabs(target_vel_left) < 0.01) { pwm_left = 0; pidLeft.reset(); } 
    else {
        double out = target_vel_left + pidLeft.compute(target_vel_left, current_vel_left);
        pwm_left = (int)(clamp(out / VEL_MAX, -1.0, 1.0) * 255.0);
        if (pwm_left > 0 && pwm_left < MIN_PWM) pwm_left = MIN_PWM;
        if (pwm_left < 0 && pwm_left > -MIN_PWM) pwm_left = -MIN_PWM;
    }

    // RIGHT PID
    if (fabs(target_vel_right) < 0.01) { pwm_right = 0; pidRight.reset(); } 
    else {
        double out = target_vel_right + pidRight.compute(target_vel_right, current_vel_right);
        pwm_right = (int)(clamp(out / VEL_MAX, -1.0, 1.0) * 255.0);
        if (pwm_right > 0 && pwm_right < MIN_PWM) pwm_right = MIN_PWM;
        if (pwm_right < 0 && pwm_right > -MIN_PWM) pwm_right = -MIN_PWM;
    }

    // Publish Debug
    debug2_msg.x = (double)current_vel_left; debug2_msg.y = (double)target_vel_left; debug2_msg.z = (double)pwm_left;
    rcl_publish(&debug2, &debug2_msg, NULL);
    
    debug_right_msg.x = (double)current_vel_right; debug_right_msg.y = (double)target_vel_right; debug_right_msg.z = (double)pwm_right;
    rcl_publish(&debug_right, &debug_right_msg, NULL);

    // Actuate
    drive_md10c(L_DIR, L_PWM, pwm_left);
    drive_md10c(R_DIR, R_PWM, pwm_right);
  }
}