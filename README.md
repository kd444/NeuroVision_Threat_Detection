# NeuroVision Threat Detection

**Real-time threat detection with weapon recognition and detailed human identification.**

## 🚀 Overview

NeuroVision is a FastAPI-based real-time video surveillance system that leverages computer vision and AI to:

-   Detect weapons in live video feeds
-   Identify the nearest person to the weapon
-   Describe the person’s clothing (type and color)
-   Trigger alerts and save snapshots when a threat is detected

## 🛠 Features

-   🎯 Real-time weapon detection using deep learning
-   🧍 Human identification with spatial proximity analysis
-   👕 Clothing description including color and type
-   🚨 Instant alert generation with image evidence
-   📁 Snapshot storage for forensic reference

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neurovision-threat-detection.git
cd neurovision-threat-detection

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Usage

```bash
# Run the FastAPI server
uvicorn main:app --reload
```

Access the API docs at: `http://127.0.0.1:8000/docs`

To start detection, use the appropriate API endpoint or integrate with a video source in your application.

## 📸 Example Output

-   Detected weapon in frame
-   Highlighted nearest individual
-   Clothing description: "Red hoodie, blue jeans"
-   Alert triggered and image saved to `/snapshots/`

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or new features.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## 📄 License

[MIT](LICENSE)

---

Built with ❤️ using FastAPI and OpenCV.
