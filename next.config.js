/** @type {import('next').NextConfig} */
const nextConfig = {
  // Note: 'output: export' means we can't use middleware.ts (www to non-www redirects)
  // If middleware is needed, remove this line and use a hosting platform that supports Next.js middleware
  output: 'export',
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: {
    unoptimized: true,
    domains: ['images.pexels.com'],
  },
  trailingSlash: true,
};

module.exports = nextConfig;
