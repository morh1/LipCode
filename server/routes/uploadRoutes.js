const express = require('express');
const multer = require('multer');
const { exec } = require('child_process');
const path = require('path');
const router = express.Router();

// Save uploaded files to disk
const upload = multer({ dest: 'uploads/' });

router.post('/video', upload.single('video'), (req, res) => {
  console.log("📩 Upload endpoint hit");
  const uploadedPath = req.file.path;

  // Run Docker container with mounted file
  const command = `docker run --rm -v "${path.resolve(__dirname, '../..')}/uploads:/app/uploads" lipnet-app python run_lipnet_inference.py uploads/${req.file.filename}`;

  exec(command, (err, stdout, stderr) => {
    if (err) {
      console.error(stderr);
      return res.status(500).json({ error: 'Docker inference failed' });
    }
    const lines = stdout.split('\n');
    const prediction = lines.find(line => line.startsWith('✅ Predicted Transcript:')) || 'Transcript not found';
    res.json({ transcript: prediction.replace('✅ Predicted Transcript:', '').trim() });
  });
});

module.exports = router;
