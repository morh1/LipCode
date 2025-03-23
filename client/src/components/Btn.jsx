import React from 'react';

export const Btn = ({ label, icon, onClick }) => {
    return (
        <div
            onClick={onClick}
            className="flex relative justify-center items-center w-[226px] h-[56px] rounded-[10px] max-md:w-[200px] max-md:h-[50px] cursor-pointer bg-[#261046]">
            <div className="flex relative justify-center items-center h-full w-full gap-3.5">
                {icon && (
                    <img
                        loading="lazy"
                        src={icon}
                        className="object-contain w-[24px] h-[24px]"
                        alt={`${label} icon`}
                    />
                )}
                <div className="self-stretch my-auto text-white text-sm">{label}</div>
            </div>
        </div>
    );
};
