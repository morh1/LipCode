"use client";
import * as React from "react";
import { Logo } from "../Logo.jsx";
import UploadPlaceholder from "./UploadPlaceholder";
import signbtnImage from '/src/assets/signbtn.png';
import { Btn } from '../Btn.jsx';

const UploadPage = () => {
<<<<<<< Updated upstream
  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    console.log("Selected file:", file);
=======
  const [file, setFile] = useState(null);
  const [transcript, setTranscript] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [transcriptAReady, setTranscriptAReady] = useState(false);
  const [transcriptBReady, setTranscriptBReady] = useState(false);

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setTranscript("");
      setTranscriptAReady(false);
      setTranscriptBReady(false);
    }
  };

  const triggerFileInput = () => {
    document.getElementById("video-upload").click();
  };

  const handleConvertClick = async () => {
    if (!file) {
      setError("Please upload a video first.");
      return;
    }

    setLoading(true);
    setError("");
    setTranscript("");
    setTranscriptAReady(false);
    setTranscriptBReady(false);

    try {
      const formData = new FormData();
      formData.append("video", file);

      const response = await fetch("http://localhost:5000/api/upload/video", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();
      const result = data.transcript || "No transcript returned.";
      setTranscript(result);
      setTranscriptAReady(true);
    } catch (err) {
      console.error("❌ Upload or inference failed:", err);
      setError("An error occurred during upload or processing.");
    } finally {
      setLoading(false);
    }
  };

  const handleConvertClickB = async () => {
    if (!file) {
      setError("Please upload a video first.");
      return;
    }

    setLoading(true);
    setError("");
    setTranscript("");
    setTranscriptAReady(false);
    setTranscriptBReady(false);

    try {
      const formData = new FormData();
      formData.append("video", file);

      const response = await fetch("http://localhost:5000/api/upload/videoB", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();
      const result = data.transcript || "No transcript returned.";
      setTranscript(result);
      setTranscriptBReady(true);
    } catch (err) {
      console.error("❌ Model B upload or inference failed:", err);
      setError("An error occurred during Model B processing.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = () => {
    const doc = new jsPDF();

    const margin = 10;
    const pageHeight = doc.internal.pageSize.height;
    const lineHeight = 10;
    let y = margin;

    const lines = doc.splitTextToSize(transcript, 180);

    lines.forEach((line) => {
      if (y + lineHeight > pageHeight - margin) {
        doc.addPage();
        y = margin;
      }
      doc.text(line, margin, y);
      y += lineHeight;
    });

    doc.save("transcript.pdf");
>>>>>>> Stashed changes
  };

  return (
    <section className="overflow-hidden bg-slate-950 h-screen relative">
      <img
        src="https://cdn.builder.io/api/v1/image/assets/16cae08561534f72919fa5995201a8ed/236a7bf1d2b3394ddd7bd4e9a3afbe9930a079ab?placeholderIfAbsent=true"
        alt="Background"
        className="object-cover absolute inset-0 w-full h-full"
      />

      <div className="flex flex-col items-start px-11 pt-8 pb-14 w-full rounded-2xl h-full max-md:px-5 max-md:max-w-full relative z-10">
        <Logo />

        <h1 className="relative mt-7 ml-16 text-7xl font-bold text-white max-md:ml-5 max-md:text-4xl">
          Upload your video
        </h1>

        <h2 className="relative mt-6 ml-16 text-2xl font-bold text-white max-md:ml-5">
          LipCode will recognized your words and generate a transcript
        </h2>

        <UploadPlaceholder onFileSelect={handleFileSelect} />
        <button
            type="button"
<<<<<<< Updated upstream
            className="relative flex justify-center items-center mx-auto w-[250px] h-[50px] mt-6 whitespace-nowrap rounded-xl cursor-pointer"
=======
            onClick={triggerFileInput}
            className="bg-white text-black px-12 py-3 rounded-xl font-semibold text-lg hover:bg-gray-200 transition-all"
          >
            {file ? `✅ Video Uploaded: ${file.name}` : "📤 Choose a Video"}
          </button>

          <input
            id="video-upload"
            type="file"
            accept="video/*"
            className="hidden"
            onChange={handleFileSelect}
          />

          {/* Model A Button */}
          <button
            type="button"
            onClick={transcriptAReady ? handleDownloadPDF : handleConvertClick}
            disabled={!file}
            className={`relative flex justify-center items-center w-[250px] h-[50px] whitespace-nowrap rounded-xl cursor-pointer ${
              !file ? "opacity-50 cursor-not-allowed" : ""
            }`}
>>>>>>> Stashed changes
            style={{
              backgroundImage: `url(${signbtnImage})`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
            }}
          >
<<<<<<< Updated upstream
            <span className="text-white text-2xl font-bold z-10">convert to text</span>
          </button>
=======
            <span className="text-white text-2xl font-bold z-10">
              {transcriptAReady ? "Download PDF" : "Model A"}
            </span>
          </button>

          {/* Model B Button */}
          <button
            type="button"
            onClick={transcriptBReady ? handleDownloadPDF : handleConvertClickB}
            disabled={!file}
            className={`relative flex justify-center items-center w-[250px] h-[50px] whitespace-nowrap rounded-xl cursor-pointer ${
              !file ? "opacity-50 cursor-not-allowed" : ""
            }`}
            style={{
              backgroundImage: `url(${signbtnImage})`,
              backgroundSize: "cover",
              backgroundPosition: "center",
            }}
          >
            <span className="text-white text-2xl font-bold z-10">
              {transcriptBReady ? "Download PDF" : "Model B"}
            </span>
          </button>
        </div>

        <div className="p-6 mt-10 mx-auto text-center flex flex-col items-center space-y-6">
          {loading && (
            <p className="text-yellow-300 text-xl text-center mt-6">
              ⏳ Processing video...
            </p>
          )}

          {transcript && (
            <p className="text-green-300 text-xl text-center mt-6">
              📝 {transcript}
            </p>
          )}

          {error && (
            <p className="text-red-500 text-xl text-center mt-6">{error}</p>
          )}
        </div>
>>>>>>> Stashed changes
      </div>
    </section>
  );
};

export default UploadPage;
