const express = require('express');
const multer = require('multer');
const { exec } = require('node:child_process');
const fs = require('fs');
const path = require('path');
const router = express.Router();

const upload = multer({ dest: 'uploads/' });

router.post('/video', upload.single('video'), (req, res) => {
  console.log("📩 Upload endpoint hit");

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

module.exports = router;
