/* SPDX-License-Identifier: GPL-3.0-or-later */

#include "language_manager.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QIODevice>
#include <QQmlComponent>
#include <QQmlEngine>
#include <QSettings>
#include <QString>
#include <QTemporaryDir>
#include <QUrl>
#include <QVariant>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

using lsfgvk::ui::LanguageManager;

namespace {

    void expect(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    QByteArray readAll(const QString& path) {
        QFile file(path);
        expect(file.open(QIODevice::ReadOnly), "unable to read test file");
        return file.readAll();
    }

    void expectLabel(QObject* object, const char* expected) {
        expect(object->property("label").toString() == QString::fromUtf8(expected),
            "QML translation binding has the wrong text");
    }

}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Expected the generated translation directory.\n";
        return 1;
    }

    QTemporaryDir settingsDirectory;
    if (!settingsDirectory.isValid()) {
        std::cerr << "Unable to create temporary settings directory.\n";
        return 1;
    }

    qputenv("XDG_CONFIG_HOME", settingsDirectory.path().toUtf8());
    QCoreApplication app(argc, argv);
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(
        QSettings::IniFormat,
        QSettings::UserScope,
        settingsDirectory.path());

    try {
        QSettings isolatedSettings{
            QStringLiteral("lsfg-vk"), QStringLiteral("lsfg-vk-ui")};
        isolatedSettings.clear();
        isolatedSettings.sync();
        expect(QFileInfo(isolatedSettings.fileName()).absoluteFilePath()
                .startsWith(settingsDirectory.path()),
            "QSettings escaped the temporary test directory");

        const auto configDirectory = settingsDirectory.filePath("lsfg-vk");
        expect(QDir{}.mkpath(configDirectory), "unable to create test config directory");
        const auto configPath = QDir(configDirectory).filePath("conf.toml");
        const QByteArray configContents = R"(version = 2
[global]
dll = "/test/Lossless.dll"
allow_fp16 = true

[[profile]]
name = "i18n-regression"
multiplier = 3
flow_scale = 0.75
performance_mode = true
pacing = "none"
gpu = "1002:73bf"
frame_generation_mode = "fixed"
target_fps = 144
active_in = ["game.exe"]
)";
        {
            QFile config(configPath);
            expect(config.open(QIODevice::WriteOnly | QIODevice::Truncate),
                "unable to create test config");
            expect(config.write(configContents) == configContents.size(),
                "unable to write complete test config");
        }

        expect(LanguageManager::effectiveLanguage(
                LanguageManager::System, QLocale{QStringLiteral("pt_PT")})
                == LanguageManager::PortugueseBrazil,
            "Portuguese locale did not resolve to pt_BR");
        expect(LanguageManager::effectiveLanguage(
                LanguageManager::System, QLocale{QStringLiteral("es_MX")})
                == LanguageManager::Spanish,
            "Spanish locale did not resolve to es");
        expect(LanguageManager::effectiveLanguage(
                LanguageManager::System, QLocale{QStringLiteral("en_US")})
                == LanguageManager::English,
            "English locale did not resolve to English");
        expect(LanguageManager::effectiveLanguage(
                LanguageManager::System, QLocale::c())
                == LanguageManager::English,
            "unsupported locale did not fall back to English");

        {
            QQmlEngine engine;
            LanguageManager manager{engine, QString::fromLocal8Bit(argv[1])};
            expect(manager.currentLanguage() == LanguageManager::System,
                "first run did not select System Default");

            QQmlComponent component(&engine);
            component.setData(
                "import QtQml\nQtObject { property string label: qsTr(\"Profiles\") }",
                QUrl{QStringLiteral("qrc:/UI.qml")});
            std::unique_ptr<QObject> translatedObject{component.create()};
            expect(translatedObject != nullptr, "unable to create QML translation probe");

            manager.setCurrentLanguage(LanguageManager::English);
            expectLabel(translatedObject.get(), "Profiles");
            expect(QCoreApplication::translate("UI", "Profiles")
                    == QStringLiteral("Profiles"),
                "English fallback unexpectedly used a translated label");

            manager.setCurrentLanguage(LanguageManager::PortugueseBrazil);
            expectLabel(translatedObject.get(), "Perfis");
            expect(QCoreApplication::translate("UI", "Default")
                    == QString::fromUtf8("Padrão"),
                "pt_BR catalog was not loaded");

            manager.setCurrentLanguage(LanguageManager::Spanish);
            expectLabel(translatedObject.get(), "Perfiles");
            expect(QCoreApplication::translate("UI", "Default")
                    == QStringLiteral("Predeterminado"),
                "Spanish catalog was not loaded");

            manager.setCurrentLanguage(LanguageManager::English);
            expectLabel(translatedObject.get(), "Profiles");
            expect(isolatedSettings.value(QStringLiteral("ui/language")).toString()
                    == QStringLiteral("en"),
                "manual language choice was not persisted");
        }

        {
            QQmlEngine engine;
            LanguageManager restoredManager{engine, QString::fromLocal8Bit(argv[1])};
            expect(restoredManager.currentLanguage() == LanguageManager::English,
                "persisted language choice was not restored");
        }

        expect(readAll(configPath) == configContents,
            "language switching modified conf.toml");

        std::cout << "All UI language manager tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << '\n';
        return 1;
    }
}
