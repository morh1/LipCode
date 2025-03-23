import React from 'react';
import logoImg from '/src/assets/logo.png';

export const Logo = () => {
    return (
        <div className="flex relative gap-1 text-2xl whitespace-nowrap">
            <img
                loading="lazy"
                src={logoImg}
                className="object-contain shrink-0 my-auto aspect-[-5] w-[160px]"
                alt="LipCode company logo"
            />
        </div>
    );
};