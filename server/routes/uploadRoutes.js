const express = require('express');
const multer = require('multer');
const { exec } = require('node:child_process');
const fs = require('fs');
const path = require('path');

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

// === Route 1: /video — LIPNET ===
router.post('/video', upload.single('video'), (req, res) => {
  console.log("📩 [LipNet] Upload endpoint hit");

  const ext = path.extname(req.file.originalname) || '.mpg';
  const tempPath = req.file.path;
  const finalName = req.file.filename + ext;
  const finalPath = path.join(path.dirname(tempPath), finalName);

  fs.rename(tempPath, finalPath, (renameErr) => {
    if (renameErr) return res.status(500).json({ error: 'Rename failed' });

    console.log("📄 Uploaded file path:", req.file.path);
    console.log("📄 Original filename:", req.file.originalname);
    console.log("📄 Temp filename:", req.file.filename);
    console.log("📄 Final name:", finalName);
    console.log("📄 Final path:", finalPath);

    const dockerSafeFilename = finalName.replace(/\\/g, '/');
    const uploadFolder = path.resolve(__dirname, '../uploads');
    const dockerCommand = `docker run --rm -v "${uploadFolder.replace(/\\/g, '/')}:/app/uploads" lipnet-app /bin/bash -c "ffmpeg -i uploads/${dockerSafeFilename} -y -an -q:v 5 -f mpeg uploads/converted.mpg && python run_lipnet_inference.py uploads/converted.mpg"`;

    console.log("🐳 Docker command:", dockerCommand);

    const waitUntilReady = () => {
      fs.access(finalPath, fs.constants.R_OK, (accessErr) => {
        if (accessErr) return setTimeout(waitUntilReady, 100);

        exec(dockerCommand, (dockerErr, dockerOut, dockerErrOut) => {
          if (dockerErr) {
            console.error("🐳 Docker stderr:", dockerErrOut);
            return res.status(500).json({ error: 'Docker inference failed' });
          }

          console.log("🪵 Full Docker output:\n", dockerOut);
          const lines = dockerOut.split('\n');
          const transcriptLine = lines.find(line => line.includes('Predicted Transcript:'));
          const transcript = transcriptLine
            ? transcriptLine.split(':').slice(1).join(':').trim()
            : 'Transcript not found';

          res.json({ transcript });
        });
      });
    };

    waitUntilReady();
  });
});


// === Route 2: /videoB — MODEL B ===
// Define the upload directory for Model B
const uploadModelBDir = path.resolve(__dirname, '../uploadModelB');

// Create the directory if it doesn't exist
if (!fs.existsSync(uploadModelBDir)) {
  fs.mkdirSync(uploadModelBDir);
}

// Configure Multer to store files directly in uploadModelB/ with original extension
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadModelBDir);
  },
  filename: (req, file, cb) => {
    // Generate unique filename using timestamp + random number
    const ext = path.extname(file.originalname) || '.mpg';
    const uniqueName = Date.now() + '-' + Math.round(Math.random() * 1e9) + ext;
    cb(null, uniqueName);
  }
});
const uploadB = multer({ storage });

// POST route to handle video upload and trigger Model B
router.post('/videoB', uploadB.single('video'), (req, res) => {
  console.log("📩 [Model B] Upload received");

  // Full path to uploaded file
  const finalPath = req.file.path;
  const finalName = req.file.filename;

  console.log("📄 [Model B] File saved to uploadModelB/:", finalPath);

  // Folder inside the Docker container to mount for transcript
  const processedPath = path.resolve(__dirname, '../../modelB/processed');
  const transcriptsPath = path.resolve(__dirname, '../uploadModelB');


  // Get the base name (without extension) to locate transcript file later
  const base = path.parse(finalName).name;

  // Construct Docker command to run Model B
  const dockerCommand = `docker run --rm \
    -v "${uploadModelBDir}:/app/uploads" \
    -v "${processedPath}:/app/processed" \
    lip-model-b \
  python /app/modelB.py --video /app/uploads/${finalName}`;

  console.log("🐳 [Model B] Running Docker command:\n", dockerCommand);

  // Execute Docker command
  exec(dockerCommand, (error, stdout, stderr) => {
    console.log("morrrrrrrrrrrrrrrrrr");
    console.log("model B stdout:", stdout);
    console.log("model B stderr:", stderr);
    if (error) {
      console.warn("Docker error:", error);
    }

    // Read the output transcript file
    const transcriptPath = path.join(transcriptsPath, `${base}_transcript.json`);
    fs.readFile(transcriptPath, 'utf-8', (readErr, transcriptData) => {
      if (readErr) {
        console.error("❌ Transcript read error:", readErr);
        return res.status(500).json({ error: 'Transcript not found' });
      }
      try {
        const transcriptJson = JSON.parse(transcriptData);
        res.json({ transcript: transcriptJson.transcript || "Transcript missing" });
      } catch (jsonErr) {
        console.error("❌ JSON parse error:", jsonErr);
        res.status(500).json({ error: 'Transcript JSON parse error' });
      }
    });
  });
});

module.exports = router;
