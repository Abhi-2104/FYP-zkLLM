# VerifAI - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### 1. Install Dependencies
```bash
cd /home/abhi/DISK2/FYP/zkllm_v2/user_interface
pip install -r requirements.txt
```

### 2. Start the Server
```bash
./start.sh
```

Or manually:
```bash
python3 app.py
```

### 3. Open Your Browser
Navigate to: **http://localhost:5000**

---

## 📱 User Interfaces

### 1. **Dashboard** (http://localhost:5000)
- Overview of the system
- Model information
- Quick access to all portals

### 2. **Provider Portal** (http://localhost:5000/provider)
- View uploaded commitments
- Model deployment status
- Cryptographic details

### 3. **User Interface** (http://localhost:5000/user)
- **Submit prompts** to the LLM
- **Generate proofs** for layers 0 and 1
  - RMSNorm (19 polynomials)
  - Self-Attention (68 polynomials)
- **Real-time progress tracking**
- **Verify proofs** with one click

### 4. **Verifier** (http://localhost:5000/verifier)
- Independent proof verification
- Component-level breakdown
- Security testing tools

---

## 🎯 Typical Workflow

### For Users:
1. Go to **User Interface** page
2. Enter a prompt (or use the default)
3. Click "Submit Prompt" (captures activations)
4. For each layer (0 and 1):
   - Click "Generate Proof" for RMSNorm
   - Wait for progress bar to complete
   - Click "Verify Proof"
   - Repeat for Self-Attention

### For Verifiers:
1. Go to **Verifier** page
2. Click "Verify RMSNorm Proof" for each layer
3. Click "Verify Self-Attention Proof"
4. View detailed component verification

---

## 🔧 Features

### Real-time Progress
- ✅ Live progress bars with percentages
- ✅ Step-by-step status messages
- ✅ WebSocket updates

### Component Breakdown
- ✅ RMSNorm: 19 polynomials
- ✅ Self-Attention: 68 polynomials
  - Q projection (12)
  - K projection (12)
  - V projection (12)
  - Q @ K^T scores (12)
  - Softmax (1)
  - Pooling (7)
  - O projection (12)

### Modern UI
- ✅ Animated backgrounds
- ✅ Smooth transitions
- ✅ Responsive design
- ✅ Dark theme
- ✅ Status badges with live dots

---

## 📊 What Happens Behind the Scenes

### Proof Generation:
```
User clicks "Generate Proof"
    ↓
Backend runs: ./rmsnorm_v2 [args] or ./self-attn_v2 [args]
    ↓
Progress updates sent via WebSocket
    ↓
Proof saved to zkllm-workdir/Llama-2-7b/<session_id>/
    ↓
"Verify" button enabled
```

### Verification:
```
User clicks "Verify Proof"
    ↓
Backend runs: ./verify_rmsnorm_v2 [args] or ./verify_self-attn_v2 [args]
    ↓
Status updates sent via WebSocket
    ↓
Component results displayed
    ↓
Success/failure notification
```

---

## 🎨 UI Components

### Cards
Each major section is a card with:
- Animated hover effects
- Gradient borders
- Smooth shadows

### Progress Bars
- Width animates based on percentage
- Shimmer effect during progress
- Color-coded status

### Status Badges
- Success (green)
- Running (orange)
- Failed (red)
- Pending (gray)
- Animated blinking dots

### Buttons
- Primary (gradient blue/purple)
- Success (green)
- Secondary (gray)
- Hover effects with lift animation
- Disabled state handling

---

## 🔐 Security Features

### Cryptographic Verification
- All proofs use BLS12-381 elliptic curve
- Sumcheck protocol for polynomial verification
- 128-bit security level

### Attack Detection
- Cross-layer proof verification fails
- Commitment binding prevents weight substitution
- Zero-knowledge property maintained

---

## 🛠️ Troubleshooting

### Server won't start:
```bash
# Check if port 5000 is available
lsof -i :5000
# Kill process if needed
kill -9 <PID>
```

### WebSocket not connecting:
- Refresh the browser
- Check browser console for errors
- Ensure Flask-SocketIO is installed

### Proof generation fails:
- Check that activation files exist in `activations/`
- Verify binary files are compiled (`rmsnorm_v2`, `self-attn_v2`)
- Check terminal for error messages

### Verification button disabled:
- Ensure proof was generated successfully
- Check proof file exists in `zkllm-workdir/Llama-2-7b/<session_id>/`
- Reload the page to refresh status

---

## 📝 Notes

- **Layers**: Currently configured for layers 0 and 1
- **Model**: Llama-2-7b (can be changed in `app.py`)
- **Sequence Length**: 128 tokens
- **Hidden Size**: 4096

To add more layers, modify the templates to include additional layer cards.

---

## 🎉 Enjoy Using VerifAI!

For more details, see the full [README.md](README.md)
