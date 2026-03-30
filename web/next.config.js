/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy API calls to FastAPI backend during dev
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ];
  },
};

export default nextConfig;
