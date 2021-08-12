#pragma once
#include <Media/Commands/ChangeAudioFile.hpp>
#include <Media/Sound/SoundModel.hpp>
#include <Process/OfflineAction/OfflineAction.hpp>

#include <score/command/Dispatchers/CommandDispatcher.hpp>
#include <score/document/DocumentContext.hpp>

#include <core/document/Document.hpp>

#include <ossia/audio/drwav_handle.hpp>

#include <QDir>
namespace AudioOracleMFCC
{

class GainDoubler final : public Process::OfflineAction
{
  SCORE_CONCRETE("7FD45402-CCE1-484C-AF9E-20CFEC648ECF")

  QString title() const noexcept override;
  UuidKey<Process::ProcessModel> target() const noexcept override;
  void
  apply(Process::ProcessModel& proc, const score::DocumentContext&) override;
};
}