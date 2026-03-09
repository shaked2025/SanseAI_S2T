# RMS Threshold Fix - "Too Quiet" Issue

## 🔍 **Problem**

User reported that the system says speech is "too quiet" even when speaking normally.

## 📊 **Root Cause**

The RMS thresholds were set too high based on calibration data:
- **Speech detection threshold**: 699 (too high)
- **Quality validator min_rms**: 500 (too high)

These thresholds were based on calibration data with median RMS of 1555, but may not match the user's current microphone setup or speaking volume.

## ✅ **Fixes Applied**

### **1. Lowered Speech Detection RMS Threshold**
- **Before**: 699 (45% of median 1555)
- **After**: 400 (more lenient, ~26% of median)
- **Location**: `main_comprehensive.py` - speech detection loop
- **Impact**: System will now detect speech at lower volumes

### **2. Lowered Quality Validator RMS Threshold**
- **Before**: 500
- **After**: 300
- **Location**: `enhanced_enrollment_quality.py` - EnrollmentQualityValidator
- **Impact**: Enrollment samples won't be rejected for being "too quiet" as easily

### **3. Relaxed Other Quality Checks**
- **SNR threshold**: 12.0 dB → 10.0 dB (more lenient)
- **Duration**: 3.0s → 2.5s (accepts shorter samples)
- **Quality penalty**: 0.3 → 0.5 (less harsh penalty for low RMS)

### **4. Added Debug Output**
- Shows actual RMS value during enrollment
- Shows threshold being used
- Helps diagnose if thresholds need further adjustment

## 🎯 **New Thresholds**

| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| Speech Detection RMS | 699 | 400 | -43% |
| Quality Validator Min RMS | 500 | 300 | -40% |
| SNR Threshold | 12.0 dB | 10.0 dB | -17% |
| Min Duration | 3.0s | 2.5s | -17% |

## 📈 **Expected Behavior**

- **Before**: Normal speech might be rejected as "too quiet"
- **After**: Normal speech should be detected and accepted
- **Trade-off**: May accept some quieter background noise, but should work better for normal speech

## 🔧 **If Still Too Strict**

If the system still says "too quiet" with normal speech, you can:

1. **Check actual RMS values**: The debug output will show what RMS is being detected
2. **Further lower thresholds**: Can reduce to 300 (speech) and 200 (quality) if needed
3. **Check microphone settings**: Ensure microphone volume/gain is set appropriately

## 💡 **Recommendation**

The thresholds are now more lenient. If you still see "too quiet" messages:
- Check the debug output to see actual RMS values
- If RMS is consistently below 300, consider:
  - Increasing microphone gain/volume
  - Speaking closer to microphone
  - Checking microphone hardware

---

**The system should now accept normal speech volume much better!** 🎯

