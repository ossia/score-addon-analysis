cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

if(NOT TARGET score_lib_base)
  include("${SCORE_SOURCE_DIR}/cmake/ScoreExternalAddon.cmake")
endif()

project(score_plugin_analysis LANGUAGES CXX)

if(NOT "${OSSIA_FFT}" STREQUAL "DOUBLE")
  message(FATAL_ERROR "Analysis needs FFTW (double)")
  return()
endif()

add_library(
  score_plugin_analysis
  Analysis/Centroid.hpp
  Analysis/ComplexSpectralDifference.hpp
  Analysis/Crest.hpp
  Analysis/EnergyDifference.hpp
  Analysis/Envelope.hpp
  Analysis/Flatness.hpp
  Analysis/GistState.hpp
  Analysis/HighFrequencyContent.hpp
  Analysis/Kurtosis.hpp
  Analysis/MFCC.hpp
  Analysis/Pitch.hpp
  Analysis/Rolloff.hpp
  Analysis/SpectralDifference.hpp
  Analysis/SpectralDifference_HWR.hpp
  Analysis/ZeroCrossing.hpp

  score_plugin_analysis.hpp
  score_plugin_analysis.cpp

  3rdparty/Gist/src/CoreFrequencyDomainFeatures.cpp
  3rdparty/Gist/src/CoreFrequencyDomainFeatures.h
  3rdparty/Gist/src/CoreTimeDomainFeatures.cpp
  3rdparty/Gist/src/CoreTimeDomainFeatures.h
  3rdparty/Gist/src/Gist.cpp
  3rdparty/Gist/src/Gist.h
  3rdparty/Gist/src/MFCC.cpp
  3rdparty/Gist/src/MFCC.h
  3rdparty/Gist/src/OnsetDetectionFunction.cpp
  3rdparty/Gist/src/OnsetDetectionFunction.h
  3rdparty/Gist/src/WindowFunctions.cpp
  3rdparty/Gist/src/WindowFunctions.h
  3rdparty/Gist/src/Yin.cpp
  3rdparty/Gist/src/Yin.h

  AudioOracle/AudioOracleFactory.cpp
  AudioOracle/AudioOracleFactory.hpp
  AudioOracle/AudioOracleMFCCFactory.cpp
  AudioOracle/AudioOracleMFCCFactory.hpp
  AudioOracle/AudioOraclePitchFactory.cpp
  AudioOracle/AudioOraclePitchFactory.hpp
  AudioOracle/AudioOracleSpectralCentroidFactory.cpp
  AudioOracle/AudioOracleSpectralCentroidFactory.hpp
  AudioOracle/AudioOracleZCSFactory.cpp
  AudioOracle/AudioOracleZCSFactory.hpp
  AudioOracle/AudioOracleRMSFactory.cpp
  AudioOracle/AudioOracleRMSFactory.hpp

  VariableMarkovOracle/VariableMarkovOracleFactory.cpp
  VariableMarkovOracle/VariableMarkovOracleFactory.hpp
  VariableMarkovOracle/VariableMarkovOracleMFCCFactory.cpp
  VariableMarkovOracle/VariableMarkovOracleMFCCFactory.hpp
  VariableMarkovOracle/VariableMarkovOraclePitchFactory.cpp
  VariableMarkovOracle/VariableMarkovOraclePitchFactory.hpp
  VariableMarkovOracle/VariableMarkovOracleCentroidFactory.cpp
  VariableMarkovOracle/VariableMarkovOracleCentroidFactory.hpp
  VariableMarkovOracle/VariableMarkovOracleZCSFactory.cpp
  VariableMarkovOracle/VariableMarkovOracleZCSFactory.hpp
  VariableMarkovOracle/VariableMarkovOracleRMSFactory.cpp
  VariableMarkovOracle/VariableMarkovOracleRMSFactory.hpp
)

target_compile_definitions(score_plugin_analysis PRIVATE USE_OSSIA_FFT=1)
target_include_directories(score_plugin_analysis PRIVATE 3rdparty/Gist/src 3rdparty/)
target_link_libraries(score_plugin_analysis PUBLIC score_plugin_engine score_plugin_media ${FFTW3_LIBRARY})

setup_score_plugin(score_plugin_analysis)


