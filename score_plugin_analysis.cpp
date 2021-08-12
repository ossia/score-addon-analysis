// This is an open source non-commercial project. Dear PVS-Studio, please check
// it. PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include "score_plugin_analysis.hpp"

#include <score/plugins/FactorySetup.hpp>

#include <Engine/Node/SimpleApi.hpp>
#include <Analysis/Centroid.hpp>
#include <Analysis/ComplexSpectralDifference.hpp>
#include <Analysis/Crest.hpp>
#include <Analysis/EnergyDifference.hpp>
#include <Analysis/Flatness.hpp>
#include <Analysis/HighFrequencyContent.hpp>
#include <Analysis/Kurtosis.hpp>
#include <Analysis/MFCC.hpp>
#include <Analysis/Pitch.hpp>
#include <Analysis/Rolloff.hpp>
#include <Analysis/SpectralDifference.hpp>
#include <Analysis/SpectralDifference_HWR.hpp>
#include <Analysis/Envelope.hpp>
#include <Analysis/ZeroCrossing.hpp>

#include <AudioOracle/AudioOracleFactory.hpp>
#include <AudioOracle/AudioOracleMFCCFactory.hpp>
#include <AudioOracle/AudioOracleRMSFactory.hpp>
#include <AudioOracle/AudioOraclePitchFactory.hpp>
#include <AudioOracle/AudioOracleSpectralCentroidFactory.hpp>
#include <AudioOracle/AudioOracleZCSFactory.hpp>

#include <VariableMarkovOracle/VariableMarkovOracleFactory.hpp>
#include <VariableMarkovOracle/VariableMarkovOracleMFCCFactory.hpp>
#include <VariableMarkovOracle/VariableMarkovOracleRMSFactory.hpp>
#include <VariableMarkovOracle/VariableMarkovOraclePitchFactory.hpp>
#include <VariableMarkovOracle/VariableMarkovOracleCentroidFactory.hpp>
#include <VariableMarkovOracle/VariableMarkovOracleZCSFactory.hpp>

#include <score_plugin_engine.hpp>

score_plugin_analysis::score_plugin_analysis() = default;
score_plugin_analysis::~score_plugin_analysis() = default;

std::vector<std::unique_ptr<score::InterfaceBase>> score_plugin_analysis::factories(
    const score::ApplicationContext& ctx,
    const score::InterfaceKey& key) const
{
  if(key == Process::OfflineAction::static_interfaceKey())
  {
    return instantiate_factories<
        score::ApplicationContext,
        FW<Process::OfflineAction,
           AudioOracle::GainDoubler,
           AudioOracleMFCC::GainDoubler,
           AudioOracleRMS::GainDoubler,
           AudioOraclePitch::GainDoubler,
           AudioOracleSpectralCentroid::GainDoubler,
           AudioOracleZCS::GainDoubler,
           VariableMarkovOracle::GainDoubler,
           VariableMarkovOracleMFCC::GainDoubler,
           VariableMarkovOracleRMS::GainDoubler,
           VariableMarkovOraclePitch::GainDoubler,
           VariableMarkovOracleCentroid::GainDoubler,
           VariableMarkovOracleZCS::GainDoubler
        >>(ctx, key);
  }
  else
  {
  return Control::instantiate_fx<
      Analysis::Centroid,
      Analysis::CSD,
      Analysis::Crest,
      Analysis::EnergyDifference,
      Analysis::Flatness,
      Analysis::Hfq,
      Analysis::Kurtosis,
      Analysis::MelSpectrum,
      Analysis::MFCC,
      Analysis::Peak,
      Analysis::Pitch,
      Analysis::RMS,
      Analysis::Rolloff,
      Analysis::SpectralDiff,
      Analysis::SpectralDiffHWR,
      Analysis::Spectrum,
      Analysis::ZeroCrossing
      >(ctx, key);
  }
}

auto score_plugin_analysis::required() const -> std::vector<score::PluginKey>
{
  return {score_plugin_engine::static_key()};
}

#include <score/plugins/PluginInstances.hpp>
SCORE_EXPORT_PLUGIN(score_plugin_analysis)
