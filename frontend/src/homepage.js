import React from 'react';
import Header from './components/Header';
import Card from './components/Card';

export default function Homepage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-indigo-50/30 to-purple-50/30">
      <Header />
      
      {/* Hero Section */}
      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Main Hero Card */}
          <div className="col-span-2">
            <Card gradient={true} className="border-indigo-100">
              <div className="space-y-4">
                <div className="inline-block px-4 py-1 bg-gradient-to-r from-indigo-500 to-purple-500 text-white text-xs font-semibold rounded-full shadow-lg">
                  AI-Powered Detection
                </div>
                <h2 className="text-3xl font-bold text-gray-800 leading-tight">
                  Multi-Modal Detection System
                </h2>
                <p className="text-gray-600 text-lg leading-relaxed">
                  Upload handwriting samples, motor feature data, or gait timeseries for instant AI-powered predictions. 
                  Our advanced models analyze multiple modalities to provide comprehensive assessments.
                </p>
                <div className="flex flex-wrap gap-4 pt-2">
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <svg className="w-5 h-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    <span className="font-medium">Real-time Analysis</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <svg className="w-5 h-5 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    <span className="font-medium">Multiple Models</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <svg className="w-5 h-5 text-purple-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    <span className="font-medium">Secure Processing</span>
                  </div>
                </div>
              </div>
            </Card>
          </div>
          
          {/* Tips Card */}
          <div>
            <Card>
              <div className="space-y-4">
                <div className="flex items-center gap-2 mb-3">
                  <div className="p-2 bg-indigo-100 rounded-lg">
                    <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                  <h3 className="font-bold text-gray-800">Quick Tips</h3>
                </div>
                <ul className="space-y-3 text-sm text-gray-600">
                  <li className="flex gap-3 items-start">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center text-xs font-semibold">1</span>
                    <span>Use clean handwriting scans on plain backgrounds for best results</span>
                  </li>
                  <li className="flex gap-3 items-start">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-purple-100 text-purple-600 flex items-center justify-center text-xs font-semibold">2</span>
                    <span>Motor CSV should match the feature columns expected by the model</span>
                  </li>
                  <li className="flex gap-3 items-start">
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-pink-100 text-pink-600 flex items-center justify-center text-xs font-semibold">3</span>
                    <span>Gait files are raw .txt timeseries signals from the dataset</span>
                  </li>
                </ul>
              </div>
            </Card>
          </div>
        </div>
        
        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">
          <Card className="text-center group">
            <div className="inline-flex p-4 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl text-white mb-4 shadow-lg group-hover:scale-110 transition-transform duration-300">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <h4 className="font-bold text-gray-800 mb-2 text-lg">Handwriting Analysis</h4>
            <p className="text-sm text-gray-600">Analyze handwriting patterns with advanced computer vision and deep learning models</p>
          </Card>
          
          <Card className="text-center group">
            <div className="inline-flex p-4 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl text-white mb-4 shadow-lg group-hover:scale-110 transition-transform duration-300">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h4 className="font-bold text-gray-800 mb-2 text-lg">Motor Features</h4>
            <p className="text-sm text-gray-600">Process motor assessment data via JSON or CSV with machine learning algorithms</p>
          </Card>
          
          <Card className="text-center group">
            <div className="inline-flex p-4 bg-gradient-to-br from-pink-500 to-red-600 rounded-2xl text-white mb-4 shadow-lg group-hover:scale-110 transition-transform duration-300">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h4 className="font-bold text-gray-800 mb-2 text-lg">Gait Analysis</h4>
            <p className="text-sm text-gray-600">Evaluate walking patterns from timeseries data using neural networks</p>
          </Card>
        </div>

        {/* Statistics Section */}
        <div className="mt-12">
          <Card gradient={true} className="border-purple-100">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center">
              <div>
                <div className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-2">
                  95%
                </div>
                <p className="text-sm text-gray-600 font-medium">Average Accuracy</p>
              </div>
              <div>
                <div className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-2">
                  3
                </div>
                <p className="text-sm text-gray-600 font-medium">Detection Models</p>
              </div>
              <div>
                <div className="text-4xl font-bold bg-gradient-to-r from-pink-600 to-red-600 bg-clip-text text-transparent mb-2">
                  &lt;2s
                </div>
                <p className="text-sm text-gray-600 font-medium">Analysis Time</p>
              </div>
              <div>
                <div className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-2">
                  24/7
                </div>
                <p className="text-sm text-gray-600 font-medium">System Availability</p>
              </div>
            </div>
          </Card>
        </div>

        {/* How It Works Section */}
        <div className="mt-12">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-800 mb-3">How It Works</h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Our AI-powered system uses state-of-the-art machine learning models to analyze multiple data modalities
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-indigo-100 to-indigo-200 rounded-full mb-4">
                <span className="text-2xl font-bold text-indigo-600">1</span>
              </div>
              <h4 className="font-bold text-gray-800 mb-2">Upload Data</h4>
              <p className="text-sm text-gray-600">Select and upload your handwriting, motor, or gait data files</p>
            </Card>
            
            <Card className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-purple-100 to-purple-200 rounded-full mb-4">
                <span className="text-2xl font-bold text-purple-600">2</span>
              </div>
              <h4 className="font-bold text-gray-800 mb-2">AI Processing</h4>
              <p className="text-sm text-gray-600">Our models analyze the data using advanced algorithms</p>
            </Card>
            
            <Card className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-pink-100 to-pink-200 rounded-full mb-4">
                <span className="text-2xl font-bold text-pink-600">3</span>
              </div>
              <h4 className="font-bold text-gray-800 mb-2">Get Results</h4>
              <p className="text-sm text-gray-600">Receive instant predictions and detailed analysis</p>
            </Card>
            
            <Card className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-green-100 to-green-200 rounded-full mb-4">
                <span className="text-2xl font-bold text-green-600">4</span>
              </div>
              <h4 className="font-bold text-gray-800 mb-2">Review & Export</h4>
              <p className="text-sm text-gray-600">Review results and export data for further analysis</p>
            </Card>
          </div>
        </div>

        {/* Call to Action */}
        <div className="mt-12">
          <Card gradient={true} className="text-center border-indigo-100">
            <div className="max-w-2xl mx-auto">
              <h2 className="text-2xl font-bold text-gray-800 mb-3">Ready to Get Started?</h2>
              <p className="text-gray-600 mb-6">
                Begin analyzing your data with our advanced AI models. Fast, accurate, and easy to use.
              </p>
              <div className="flex flex-wrap gap-4 justify-center">
                <a 
                  href="#handwriting" 
                  className="px-8 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white rounded-lg font-medium transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 inline-flex items-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  Start Analysis
                </a>
                <button 
                  className="px-8 py-3 bg-white hover:bg-gray-50 text-gray-700 border-2 border-gray-200 rounded-lg font-medium transition-all duration-200 shadow-md hover:shadow-lg transform hover:-translate-y-0.5 inline-flex items-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Learn More
                </button>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-white/50 backdrop-blur-sm border-t border-gray-200 mt-16 py-8">
        <div className="max-w-6xl mx-auto px-6 text-center text-sm text-gray-600">
          <p className="mb-2">
            <span className="font-semibold text-gray-800">Parkinson's Detection Models</span> - Powered by Advanced AI
          </p>
          <p className="text-xs">
            For research and educational purposes. Always consult healthcare professionals for medical advice.
          </p>
        </div>
      </footer>
    </div>
  );
}