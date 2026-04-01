# VerifAI - User Interface

Modern, interactive web interface for zero-knowledge LLM verification.

## Features

- **Provider Portal**: Upload and manage weight commitments
- **User Interface**: Prompt the LLM and generate proofs with real-time progress
- **Verifier**: Verify cryptographic proofs independently

## Installation

```bash
cd user_interface
pip install -r requirements.txt
```

## Running the Application

```bash
python app.py
```

Or use the startup script:
```bash
chmod +x start.sh
./start.sh
```

The application will be available at:
- **Main Dashboard**: http://localhost:5000
- **Provider Portal**: http://localhost:5000/provider
- **User Interface**: http://localhost:5000/user
- **Verifier**: http://localhost:5000/verifier

## Architecture

### Backend (Flask + SocketIO)
- `app.py`: Main Flask application with REST API and WebSocket support
- Real-time progress updates via SocketIO
- Background thread processing for proof generation/verification

### Frontend
- Modern, responsive UI with animations
- Real-time progress bars
- Component-based verification display
- Socket.IO client for live updates

## API Endpoints

### REST API
- `GET /`: Main dashboard
- `GET /provider`: Provider portal
- `GET /user`: User interface
- `GET /verifier`: Verification center
- `GET /api/model/info`: Get model information
- `POST /api/prompt`: Submit user prompt
- `POST /api/proof/generate`: Generate proof for layer/component
- `POST /api/verify`: Verify proof for layer/component
- `GET /api/layer/status/<layer>`: Get proof status for layer

### WebSocket Events
- `proof_progress`: Real-time proof generation progress
- `proof_complete`: Proof generation completion
- `verify_status`: Verification status updates
- `verify_complete`: Verification completion

## Proof Generation Flow

1. User submits prompt via UI
2. Activations captured (handled by backend)
3. User clicks "Generate Proof" for a component
4. Backend starts proof generation in background thread
5. Progress updates sent via WebSocket
6. Completion notification shown in UI
7. "Verify Proof" button enabled

## Verification Flow

1. User clicks "Verify Proof"
2. Backend starts verification in background thread
3. Status updates sent via WebSocket
4. Component-level verification results displayed
5. Success/failure notification shown

## Technology Stack

- **Backend**: Flask 3.0, Flask-SocketIO
- **Frontend**: Vanilla JavaScript, Socket.IO client
- **Styling**: Custom CSS with modern animations
- **Real-time**: WebSocket communication

## Directory Structure

```
user_interface/
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
├── start.sh               # Startup script
├── static/
│   ├── css/
│   │   └── main.css       # Modern UI styling
│   └── js/
│       └── main.js        # Frontend logic
└── templates/
    ├── index.html         # Dashboard
    ├── provider.html      # Provider portal
    ├── user.html          # User interface
    └── verifier.html      # Verification center
```

## Features Showcase

### Real-time Progress Tracking
- Animated progress bars
- Percentage updates
- Status messages for each step
- Component-level breakdown

### Modern UI Elements
- Gradient backgrounds with animations
- Smooth transitions and hover effects
- Status badges with animated dots
- Responsive card-based layout
- Dark theme optimized for long sessions

### Security Visualization
- Polynomial count display
- Component-level verification
- Cross-layer attack testing
- Cryptographic details panel

## Customization

### Changing Model
Edit `MODEL_CONFIG` in `app.py`:
```python
MODEL_CONFIG = {
    "name": "Your-Model-Name",
    "num_layers": 32,
    "hidden_size": 4096,
    ...
}
```

### Adding New Components
1. Add component to backend handler in `app.py`
2. Create UI card in appropriate template
3. Add progress tracking elements
4. Connect buttons to JavaScript functions

## Troubleshooting

**Port already in use:**
```bash
# Change port in app.py
socketio.run(app, host='0.0.0.0', port=5001)
```

**WebSocket connection fails:**
- Check firewall settings
- Ensure Socket.IO client version matches server
- Verify CORS settings in app.py

**Proof generation doesn't start:**
- Check binary paths in `app.py` (BASE_DIR, WORKDIR)
- Verify activation files exist
- Check terminal for error messages

## Development

To modify the UI:
1. Edit HTML templates in `templates/`
2. Update styles in `static/css/main.css`
3. Add functionality in `static/js/main.js`
4. Restart Flask server to see changes

For backend changes:
1. Modify `app.py`
2. Flask auto-reloads in debug mode
3. Check console for errors

## Production Deployment

For production use:
1. Disable debug mode in `app.py`
2. Use production WSGI server (gunicorn)
3. Set up HTTPS with proper certificates
4. Configure environment variables for secrets
5. Use Redis for SocketIO scaling

```bash
gunicorn --worker-class eventlet -w 1 app:app
```
