import React, { useState } from 'react';  // <-- Import useState!
import { Logo } from '../Logo.jsx';
import { Btn } from '../Btn.jsx';
import signbtnImage from '/src/assets/signbtn.png';

const socialButtons = [
  {
    label: 'Google',
    icon: 'https://cdn.builder.io/api/v1/image/assets/TEMP/a2d4c5f3405976e7912072d7477d71daa11d44c8f50e5a5b630f90de67f619ac?placeholderIfAbsent=true&apiKey=16cae08561534f72919fa5995201a8ed',
  },
  {
    label: 'Facebook',
    icon: 'https://cdn.builder.io/api/v1/image/assets/TEMP/4fdd9499ad810f773b3eb2f05a97e21a90b5c92f9ead3120665b5c88fe4747d2?placeholderIfAbsent=true&apiKey=16cae08561534f72919fa5995201a8ed',
  },
];

// API Function: checks if email exists on your server
const checkEmail = async (email) => {
  try {
    const response = await fetch('http://localhost:5000/api/auth/check-email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    });

    const data = await response.json();
    return data;
  } catch (error) {
    return { error: 'Server error, please try again later.' };
  }
};

const Login = () => {
  const [email, setEmail] = useState('');   // State for the email
  const [message, setMessage] = useState(''); // State for feedback messages

  // handleSubmit or handleClick—just pick one name and stay consistent
  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('Checking...');

    // Call the API function
    const data = await checkEmail(email);

    if (data.exists) {
      // If the user exists, redirect
      window.location.href = '/dashboard';
    } else {
      // Otherwise, show a message that activation email was sent
      setMessage('Activation email sent! Please check your inbox.');
    }
  };

  // This function will handle the Facebook button:
  const handleFacebookLogin = () => {
    // Typically you redirect to your server route that triggers the Facebook OAuth flow
    window.location.href = 'http://localhost:5000/api/auth/facebook';
  };

  // If you also have a Google route, you’d do something similar for Google:
  const handleGoogleLogin = () => {
    window.location.href = 'http://localhost:5000/api/auth/google';
  };

  return (
    <div className="overflow-hidden pr-20 bg-slate-900 h-screen max-md:pr-5">
      <div className="flex gap-5 max-md:flex-col">
        {/* Left Section with Logo and Intro Text */}
        <div className="flex flex-col w-[57%] max-md:ml-0 h-screen max-md:w-full justify-start">
          <div className="relative flex flex-col items-start pt-8 pr-20 pb-16 pl-10 w-full font-bold text-white h-screen max-md:px-5 max-md:mt-10 max-md:max-w-full">
            <img
              loading="lazy"
              src="https://cdn.builder.io/api/v1/image/assets/TEMP/195313fe0001def7d31f6cedc9730d3812dcc9b772aa9b451c2a34e6c5fdeae1?placeholderIfAbsent=true&apiKey=16cae08561534f72919fa5995201a8ed"
              className="absolute inset-0 w-full h-full object-left"
              alt="Background decoration"
            />
            <Logo />
            <div className="absolute bottom-8 text-4xl uppercase leading-[51px] text-white max-md:text-4xl max-md:leading-10 md:text-4xl lg:text-5xl">
              Sign in to BEGIN your<br /> adventure!
            </div>
          </div>
        </div>

        {/* Right Section with Form and Social Buttons */}
        <div className="flex flex-col ml-5 w-[35%] max-md:ml-0 max-md:w-full">
          <div className="flex flex-col self-stretch my-auto w-full text-base font-medium text-white max-md:mt-10 max-md:max-w-full">
            <div className="self-start text-7xl font-bold max-md:text-4xl lg:text-6xl">SIGN IN</div>

            {/* Use onSubmit in the form and call handleSubmit */}
            <form onSubmit={handleSubmit}>
              <label htmlFor="email" className="self-start mt-7 text-lg font-bold block">
                Sign in OR join with email address
              </label>
              <div className="flex relative bg-[#261046] flex-row gap-3.5 px-5 py-6 mt-6 whitespace-nowrap rounded-xl min-h-[69px] text-neutral-400 max-md:px-5">
                <img
                  loading="lazy"
                  src="https://cdn.builder.io/api/v1/image/assets/TEMP/bdf51c5dc12958af6b32ff32198dcde6d5151f1da84315d3727176591ef1dcdf?placeholderIfAbsent=true&apiKey=16cae08561534f72919fa5995201a8ed"
                  className="object-contain shrink-0 self-start aspect-[1.19] w-[25px]"
                  alt="Email icon"
                />
                <input
                  type="email"
                  id="email"
                  className="relative flex-auto w-[375px] bg-transparent border-none outline-none"
                  placeholder="Yourname@gmail.com"
                  aria-label="Email address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>

              {/* This button will submit the form */}
              <button
                type="submit"
                className="relative flex-center items-center w-full h-[69px] px-9 py-6 mt-6 whitespace-nowrap rounded-xl cursor-pointer"
                style={{
                  backgroundImage: `url(${signbtnImage})`,
                  backgroundSize: 'cover',
                  backgroundPosition: 'center',
                }}
              >
                <span className="text-white text-2xl font-bold z-10">CONTINUE</span>
              </button>
            </form>

            {/* Show messages to the user */}
            {message && <p className="mt-4 text-yellow-300">{message}</p>}

            <img
              loading="lazy"
              src="https://cdn.builder.io/api/v1/image/assets/TEMP/ad92a0dfd38342739bd28465ad9ec2bf829d11cde0b95d6d3565cbea303172ff?placeholderIfAbsent=true&apiKey=16cae08561534f72919fa5995201a8ed"
              className="object-contain mt-12 w-full max-md:mt-10 max-md:max-w-full"
              alt="Decorative divider"
            />
            <div className="self-start mt-12 font-semibold text-zinc-400 max-md:mt-10">
              Or continue with
            </div>

            <div className="social-buttons flex gap-2.5 mt-7 text-base font-semibold whitespace-nowrap">
              {socialButtons.map((button, index) => {
                // If button is Facebook, call handleFacebookLogin
                if (button.label === 'Facebook') {
                  return (
                    <Btn
                      key={index}
                      label={button.label}
                      icon={button.icon}
                      onClick={handleFacebookLogin}
                    />
                  );
                }
                // If button is Google, do something else
                else if (button.label === 'Google') {
                  return (
                    <Btn
                      key={index}
                      label={button.label}
                      icon={button.icon}
                      onClick={handleGoogleLogin}
                    />
                  );
                }
                // Otherwise just console.log
                else {
                  return (
                    <Btn
                      key={index}
                      label={button.label}
                      icon={button.icon}
                      onClick={() => console.log(`${button.label} clicked`)}
                    />
                  );
                }
              })}
            </div>

            <div className="self-start mt-6 text-zinc-400">
              By registering you agree with our{' '}
              <a href="#" className="text-violet-400" tabIndex={0}>
                Terms and Conditions
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
