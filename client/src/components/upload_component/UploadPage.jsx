"use client";
import * as React from "react";
import { Logo } from "../Logo.jsx";
import UploadPlaceholder from "./UploadPlaceholder";
import signbtnImage from '/src/assets/signbtn.png';
import { Btn } from '../Btn.jsx';

const UploadPage = () => {
  const [file, setFile] = React.useState(null);
  const [transcript, setTranscript] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    console.log("✅ File selected:", selectedFile.name);
    setFile(selectedFile);
    setTranscript("");
    setError("");
  };

  const handleConvertClick = async () => {
    if (!file) {
      setError("Please upload a video first.");
      return;
    }

    setLoading(true);
    setError("");
    setTranscript("");

    try {
      const formData = new FormData();
      formData.append("video", file);

      console.log("📤 Uploading video...");
      const response = await fetch("http://localhost:5000/api/upload/video", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const data = await response.json();
      console.log("✅ Inference result:", data);
      setTranscript(data.transcript || "No transcript returned.");
    } catch (err) {
      console.error("❌ Upload or inference failed:", err);
      setError("An error occurred during upload or processing.");
    } finally {
      setLoading(false);
    }
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
          LipCode will recognize your words and generate a transcript
        </h2>

        <UploadPlaceholder onFileSelect={handleFileSelect} />

        <button
          type="button"
          onClick={handleConvertClick}
          className="relative flex justify-center items-center mx-auto w-[250px] h-[50px] mt-6 whitespace-nowrap rounded-xl cursor-pointer"
          style={{
            backgroundImage: `url(${signbtnImage})`,
            backgroundSize: "cover",
            backgroundPosition: "center",
          }}
        >
          <span className="text-white text-2xl font-bold z-10">convert to text</span>
        </button>

        {loading && (
          <p className="text-yellow-300 text-xl text-center mt-4">⏳ Processing video...</p>
        )}

        {transcript && (
          <p className="text-green-300 text-xl text-center mt-4">📝 {transcript}</p>
        )}

        {error && (
          <p className="text-red-500 text-xl text-center mt-4">{error}</p>
        )}
      </div>
    </section>
  );
};

export default UploadPage;
