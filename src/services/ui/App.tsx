import React from 'react';
import CameraShooter from './components/buttons/CameraShooter';
import {Box, NativeBaseProvider} from 'native-base';

export default function App() {
    return (
        <NativeBaseProvider>
            <Box>
                <CameraShooter/>
            </Box>
        </NativeBaseProvider>
    );
}