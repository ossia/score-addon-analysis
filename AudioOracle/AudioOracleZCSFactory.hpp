#pragma once
#include <Media/Commands/ChangeAudioFile.hpp>
#include <Media/Sound/SoundModel.hpp>
#include <Process/OfflineAction/OfflineAction.hpp>

#include <score/command/Dispatchers/CommandDispatcher.hpp>
#include <score/document/DocumentContext.hpp>

#include <core/document/Document.hpp>

#include <ossia/audio/drwav_handle.hpp>

#include <QDir>
namespace AudioOracleZCS
{

class GainDoubler final : public Process::OfflineAction
{
  SCORE_CONCRETE("64F6B32A-EC94-46B3-8828-45A3D4F8A13A")

  QString title() const noexcept override;
  UuidKey<Process::ProcessModel> target() const noexcept override;
  void
  apply(Process::ProcessModel& proc, const score::DocumentContext&) override;
};
}