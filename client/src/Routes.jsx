import React from 'react';
import { useRoutes } from 'react-router-dom';
import Login from './components/login_component/Login.jsx';
import UploadPage from "./components/upload_component/UploadPage.jsx";

const ProjectRoutes = () => {
    let element = useRoutes([
        { path: '/', element: <Login /> },
        { path: '/dashboard', element: <UploadPage /> }

    ]);

    return element;
};

export default ProjectRoutes;
