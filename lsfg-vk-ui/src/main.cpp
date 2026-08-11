/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <QIcon>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QUrl>

#include "backend.hpp"

using namespace lsfgvk::ui;

int main(int argc, char* argv[]) {
    const QGuiApplication app(argc, argv);
    QGuiApplication::setWindowIcon(QIcon(":/rsc/gay.pancake.lsfg-vk-ui.png"));
    QGuiApplication::setApplicationName("lsfg-vk-ui");
    QGuiApplication::setApplicationDisplayName("lsfg-vk-ui");

    // Force a Qt Quick Controls style that honors the propagated dark palette.
    // This must happen before loading QML that imports QtQuick.Controls.
    QQuickStyle::setStyle("Fusion");

    // Backend must outlive the QML engine. Otherwise the backend QObject
    // is destroyed first during shutdown and live QML bindings briefly see a
    // null context object, producing "Cannot read property ... of null".
    Backend backend;
    QQmlApplicationEngine engine;

    engine.rootContext()->setContextProperty("backend", &backend);
    engine.load("qrc:/rsc/UI.qml");

    return QGuiApplication::exec();
}
