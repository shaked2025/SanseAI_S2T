# Microphone Setup Guide

## 🎤 Current Status

**Detected Microphone:** Logitech BRIO (camera microphone)  
**Status:** ✅ Working  
**Signal Strength:** Quiet (may need adjustment)

---

## 🔧 If You Connected a Different External Microphone:

### Set as Default Device in Windows:

**Method 1: Sound Settings (Windows 11)**
1. Right-click **speaker icon** in taskbar (bottom-right)
2. Click **"Sound settings"**
3. Scroll to **"Input"** section
4. Click dropdown and select your **external microphone**
5. Click **"Test your microphone"** - speak and see if bars move
6. If working, it's now the default!

**Method 2: Sound Control Panel (Windows 10/11)**
1. Right-click **speaker icon** in taskbar
2. Click **"Sounds"**
3. Go to **"Recording"** tab
4. Find your external microphone in the list
5. Right-click it → **"Set as Default Device"**
6. Right-click it → **"Set as Default Communication Device"**
7. Click **"OK"**

**Method 3: Check What's Available**
Run this command to see all microphones:
```bash
python test_microphone.py
```

This will show:
- All available microphones
- Which one is currently default
- Test the microphone with audio level display

---

## 🧪 Test Your External Microphone

After setting it as default:

```bash
python test_microphone.py
```

**You should see:**
- Your external microphone name
- Audio level bars when you speak
- Level should be 1500-5000 for good quality

**Good Signal:**
```
Audio Level: ████████████████████ 3500
```

**Weak Signal:**
```
Audio Level: ███ 450
```

If weak, increase microphone gain in Windows.

---

## ⚙️ Adjust Microphone Gain

**If signal is too quiet:**

1. Right-click speaker icon → **"Sounds"**
2. **"Recording"** tab
3. Double-click your microphone
4. **"Levels"** tab
5. Increase **"Microphone"** slider to 80-100
6. Increase **"Microphone Boost"** to +10 dB or +20 dB
7. Click **"OK"**
8. Test again

---

## 🎯 For Interview System

The interview system will automatically use whatever Windows has set as the **default recording device**.

After setting your external microphone as default:
1. Close the current interview app (if running)
2. Run: `python main_interview.py`
3. It will now use your external microphone!

---

## 💡 Tips

**For best results:**
- Place microphone 6-12 inches from speakers
- Quiet room (minimal background noise)
- Test before important interview
- Check levels are 1500-5000 for good quality

---

**Your Logitech BRIO is currently the default. If you want to use a different external microphone, set it as default in Windows Sound settings, then restart the interview app!**

