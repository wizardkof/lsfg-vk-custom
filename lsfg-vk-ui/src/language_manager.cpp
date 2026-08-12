/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "language_manager.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QLibraryInfo>
#include <QQmlEngine>
#include <QVariant>

#include <utility>

using namespace lsfgvk::ui;

LanguageManager::LanguageManager(
        QQmlEngine& engine,
        QString applicationTranslationPath,
        QObject* parent)
    : QObject(parent),
      m_engine(engine),
      m_application_translation_path(std::move(applicationTranslationPath)) {
    const auto storedLanguage = m_settings
        .value(QStringLiteral("ui/language"), QStringLiteral("system"))
        .toString();
    applyLanguage(languageFromCode(storedLanguage), false);
}

LanguageManager::~LanguageManager() {
    removeTranslators();
}

int LanguageManager::currentLanguage() const {
    return static_cast<int>(m_current_language);
}

void LanguageManager::setCurrentLanguage(int language) {
    if (language < static_cast<int>(System)
        || language > static_cast<int>(Spanish))
        return;

    const auto selectedLanguage = static_cast<Language>(language);
    if (selectedLanguage == m_current_language)
        return;

    applyLanguage(selectedLanguage, true);
}

LanguageManager::Language LanguageManager::effectiveLanguage(
        Language selectedLanguage,
        const QLocale& systemLocale) {
    if (selectedLanguage != System)
        return selectedLanguage;

    switch (systemLocale.language()) {
        case QLocale::Portuguese:
            return PortugueseBrazil;
        case QLocale::Spanish:
            return Spanish;
        default:
            return English;
    }
}

QString LanguageManager::languageCode(Language language) {
    switch (language) {
        case System:
            return QStringLiteral("system");
        case English:
            return QStringLiteral("en");
        case PortugueseBrazil:
            return QStringLiteral("pt_BR");
        case Spanish:
            return QStringLiteral("es");
    }
    return QStringLiteral("system");
}

LanguageManager::Language LanguageManager::languageFromCode(const QString& code) {
    if (code == QStringLiteral("en"))
        return English;
    if (code == QStringLiteral("pt_BR"))
        return PortugueseBrazil;
    if (code == QStringLiteral("es"))
        return Spanish;
    return System;
}

void LanguageManager::applyLanguage(Language selectedLanguage, bool persist) {
    removeTranslators();

    const auto resolvedLanguage = effectiveLanguage(selectedLanguage);
    const auto resolvedCode = languageCode(resolvedLanguage);

    if (resolvedLanguage != English) {
        const auto qtCatalog = QStringLiteral("qt_%1.qm").arg(resolvedCode);
        const auto qtCatalogPath = QDir(
            QLibraryInfo::path(QLibraryInfo::TranslationsPath))
            .filePath(qtCatalog);
        const auto qtCatalogLoaded = m_qt_translator.load(qtCatalogPath);

        const auto applicationCatalog =
            QStringLiteral("lsfg-vk-ui_%1.qm").arg(resolvedCode);
        const auto applicationCatalogPath = QDir(m_application_translation_path)
            .filePath(applicationCatalog);
        const auto applicationCatalogLoaded =
            m_application_translator.load(applicationCatalogPath);

        if (qtCatalogLoaded) {
            m_qt_translator_installed =
                QCoreApplication::installTranslator(&m_qt_translator);
        }
        if (applicationCatalogLoaded) {
            m_application_translator_installed =
                QCoreApplication::installTranslator(&m_application_translator);
        }
    }

    m_current_language = selectedLanguage;
    if (persist) {
        m_settings.setValue(
            QStringLiteral("ui/language"), languageCode(selectedLanguage));
        m_settings.sync();
    }

    m_engine.retranslate();
    emit currentLanguageChanged();
}

void LanguageManager::removeTranslators() {
    if (m_application_translator_installed) {
        QCoreApplication::removeTranslator(&m_application_translator);
        m_application_translator_installed = false;
    }
    if (m_qt_translator_installed) {
        QCoreApplication::removeTranslator(&m_qt_translator);
        m_qt_translator_installed = false;
    }
}
