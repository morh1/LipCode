import React from "react";

function UploadPlaceholder({ onFileSelect }) {
  return (
    <div className="flex relative flex-col justify-center items-center self-center px-6 py-16 mt-8 max-w-full rounded-3xl min-h-[400px] w-[500px] border-2 border-dashed border-gray-400 bg-black/50">
      <label htmlFor="video-upload" className="cursor-pointer">
        <img
          src="https://cdn.builder.io/api/v1/image/assets/16cae08561534f72919fa5995201a8ed/1e1ed31ca31fb519e0ba353a7cf15cb72d670a96?placeholderIfAbsent=true"
          alt="Upload Icon"
          className="object-contain max-w-full aspect-[1.14] w-[80px]"
        />
      </label>
      <input
        id="video-upload"
        type="file"
        accept="video/*"
        className="hidden"
        onChange={onFileSelect}
      />
    </div>
  );
}

export default UploadPlaceholder;
