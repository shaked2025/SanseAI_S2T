# Windows-Specific Notes

## ⚠️ Known Issue: Production Mode on Windows

### Problem
The **production speaker diarization mode** (deep learning with SpeechBrain) has a Windows compatibility issue:

**Error:** `[WinError 1314] A required privilege is not held by the client`

**Cause:** SpeechBrain tries to create symbolic links, which requires administrator privileges on Windows.

### Current Solution: Simple Mode ✅

The application now uses **SIMPLE mode** by default, which:
- ✅ Works immediately on Windows (no admin needed)
- ✅ No crashes or permission errors  
- ✅ Provides basic speaker separation with 0.82 threshold
- ⚠️ Less accurate than production mode (70-80% vs 85-95%)

### To Use Production Mode (Optional)

If you want the better accuracy of production mode, you have 3 options:

#### Option 1: Run as Administrator (Easiest)
1. Right-click on Command Prompt or PowerShell
2. Select "Run as administrator"
3. Navigate to project: `cd C:\Users\admin\SanseAI_S2T`
4. Run: `python main.py`

#### Option 2: Enable Developer Mode (Windows 10/11)
1. Open Settings → Update & Security → For Developers
2. Turn on "Developer Mode"
3. Restart your computer
4. Symlinks will now work without admin privileges

#### Option 3: Manual Model Setup
Copy the model files manually to avoid symlinks:
```bash
# Copy from cache to local directory
xcopy "%USERPROFILE%\.cache\huggingface\hub\models--speechbrain--spkrec-ecapa-voxceleb" "models\spkrec-ecapa-voxceleb" /E /I /H
```

Then edit `config.yaml`:
```yaml
diarization:
  mode: "production"
```

## Current Configuration

**Mode:** Simple (default)
**Threshold:** 0.82 (balanced)
**Status:** ✅ Working on Windows without admin

## Performance Comparison

| Feature | Simple Mode | Production Mode |
|---------|-------------|-----------------|
| **Accuracy** | 70-80% | 85-95% |
| **Windows Compatibility** | ✅ Perfect | ⚠️ Requires admin |
| **Setup** | None | Admin or Dev Mode |
| **Speed** | Fast (~50ms) | Moderate (~200ms) |
| **Features** | 5 basic | 192 deep learning |

## Recommendation

**For testing and general use:** Simple mode is fine and works out of the box.

**For production/critical use:** Enable Developer Mode or run as admin for the better accuracy.

## Testing

To verify which mode is running, check the console output:

**Simple Mode:**
```
⚡ Using SIMPLE speaker diarization (fast, basic features)
```

**Production Mode:**
```
🎯 Using PRODUCTION speaker diarization (robust, deep learning)
📂 Found cached model at: ...
✅ Speaker embedding model loaded successfully!
```

## Future Fix

We're exploring:
- Pre-downloading models without symlinks
- Using alternative embedding models
- Custom Windows-compatible installation

For now, simple mode with 0.82 threshold provides reasonable speaker separation on Windows.

---

**For most users, the current simple mode setup works well enough! 🎯**

