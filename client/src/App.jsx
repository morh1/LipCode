import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import ProjectRoutes from './Routes';

function App() {
    return (
        <BrowserRouter>
            <ProjectRoutes />
        </BrowserRouter>
    );
}

export default App;
