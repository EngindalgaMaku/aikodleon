"use client";

import Image, { type ImageProps } from 'next/image';
import { useState, useEffect } from 'react';

interface ClientImageProps extends Omit<ImageProps, 'onError'> {
  fallbackSrc: string;
}

const ClientImage: React.FC<ClientImageProps> = ({ src, fallbackSrc, alt, ...props }) => {
  const [currentSrc, setCurrentSrc] = useState(src);
  const isSvg = typeof src === 'string' && src.endsWith('.svg');

  useEffect(() => {
    setCurrentSrc(src); // Reset src if the main src prop changes
  }, [src]);

  return (
    <Image
      src={currentSrc}
      alt={alt}
      {...props}
      unoptimized={isSvg}
      onError={() => {
        if (currentSrc !== fallbackSrc) { // Prevent infinite loop if fallback also fails
          setCurrentSrc(fallbackSrc);
        }
      }}
    />
  );
};

export default ClientImage; 