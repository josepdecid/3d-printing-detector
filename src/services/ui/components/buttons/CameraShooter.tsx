import React, {MutableRefObject, useEffect, useRef, useState} from 'react';
import {Box, Fab, Flex, Text} from 'native-base';
import {Entypo} from '@expo/vector-icons';

import {Camera} from "expo-camera";
import axios from "axios";
import {Dimensions} from "react-native";


export default function CameraShooter() {
    let camera: MutableRefObject<Camera> | MutableRefObject<null> = useRef(null);
    const [hasPermission, setHasPermission] = useState(false);

    const {height, width} = Dimensions.get('window');

    useEffect(() => {
        (async () => {
            const {status} = await Camera.requestCameraPermissionsAsync();
            setHasPermission((status as string) === 'granted');
        })();
    }, []);

    const snap = async () => {
        if (camera !== null) {
            const photo = await (camera as Camera).takePictureAsync({
                quality: 0,
                base64: true
            })

            axios.post('http://192.168.43.84:8080/process_image',
                {img: photo.base64})
                .then(res => alert(res.data))
                .catch(err => console.error(err))
        }
    };

    if (hasPermission === false) {
        return <Text>No access to camera</Text>;
    }

    return (
        <Flex>
            <Camera
                ref={ref => camera = ref}
                type={Camera.Constants.Type.back}
                style={{width: width, height: `${width * 0.22}%`}}
            >
                <Box bg-white>
                    <Fab
                        onPress={() => snap()}
                        colorScheme="blue"
                        size="lg"
                        icon={<Entypo name="camera" size={42} color="white"/>}
                    />
                </Box>
            </Camera>
        </Flex>
    );
}
