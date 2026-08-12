/* SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <QLocale>
#include <QObject>
#include <QSettings>
#include <QString>
#include <QTranslator>

class QQmlEngine;

namespace lsfgvk::ui {

    class LanguageManager final : public QObject {
        Q_OBJECT
        Q_PROPERTY(int current_language READ currentLanguage WRITE setCurrentLanguage
            NOTIFY currentLanguageChanged)

    public:
        enum Language {
            System = 0,
            English,
            PortugueseBrazil,
            Spanish,
        };
        Q_ENUM(Language)

        explicit LanguageManager(
            QQmlEngine& engine,
            QString applicationTranslationPath = QStringLiteral(":/i18n"),
            QObject* parent = nullptr);
        ~LanguageManager() override;

        [[nodiscard]] int currentLanguage() const;
        void setCurrentLanguage(int language);

        [[nodiscard]] static Language effectiveLanguage(
            Language selectedLanguage,
            const QLocale& systemLocale = QLocale::system());
        [[nodiscard]] static QString languageCode(Language language);

    signals:
        void currentLanguageChanged();

    private:
        [[nodiscard]] static Language languageFromCode(const QString& code);
        void applyLanguage(Language selectedLanguage, bool persist);
        void removeTranslators();

        QQmlEngine& m_engine;
        QString m_application_translation_path;
        QSettings m_settings{QStringLiteral("lsfg-vk"), QStringLiteral("lsfg-vk-ui")};
        QTranslator m_qt_translator;
        QTranslator m_application_translator;
        Language m_current_language{System};
        bool m_qt_translator_installed{false};
        bool m_application_translator_installed{false};
    };

}
