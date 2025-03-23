"use client";
import * as React from "react";
import { Logo } from "../Logo.jsx";
import UploadPlaceholder from "./UploadPlaceholder";
import signbtnImage from '/src/assets/signbtn.png';
import { Btn } from '../Btn.jsx';

const UploadPage = () => {
  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    console.log("Selected file:", file);
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
            className="relative flex justify-center items-center mx-auto w-[250px] h-[50px] mt-6 whitespace-nowrap rounded-xl cursor-pointer"
            style={{
              backgroundImage: `url(${signbtnImage})`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
            }}
          >
            <span className="text-white text-2xl font-bold z-10">convert to text</span>
          </button>
      </div>
    </section>
  );
};

export default UploadPage;
