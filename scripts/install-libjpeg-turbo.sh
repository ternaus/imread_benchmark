#!/usr/bin/env bash
# Install the benchmark-pinned upstream libjpeg-turbo build on Debian/Ubuntu.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "install-libjpeg-turbo.sh must run as root" >&2
    exit 2
fi

LIBJPEG_TURBO_VERSION=3.2.0
case "$(dpkg --print-architecture)" in
    amd64)
        LIBJPEG_TURBO_PACKAGE="libjpeg-turbo-official_${LIBJPEG_TURBO_VERSION}_amd64.deb"
        LIBJPEG_TURBO_SHA256=21297da4a4eb34ebefc54afca5d8dd86c0fdd6a9dfe49b1b962c5d1eeeafd8ec
        ;;
    arm64)
        LIBJPEG_TURBO_PACKAGE="libjpeg-turbo-official_${LIBJPEG_TURBO_VERSION}_arm64.deb"
        LIBJPEG_TURBO_SHA256=4bf51ee5dc130052ee19ac47cc2b7b613961dcd9519398fce976bead58be36eb
        ;;
    *)
        echo "Unsupported architecture for official libjpeg-turbo: $(dpkg --print-architecture)" >&2
        exit 2
        ;;
esac

LIBJPEG_TURBO_DEB="/tmp/$LIBJPEG_TURBO_PACKAGE"
curl -LsSf \
    "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/$LIBJPEG_TURBO_VERSION/$LIBJPEG_TURBO_PACKAGE" \
    -o "$LIBJPEG_TURBO_DEB"
echo "$LIBJPEG_TURBO_SHA256  $LIBJPEG_TURBO_DEB" | sha256sum --check --strict
apt-get install -y -q "$LIBJPEG_TURBO_DEB"
printf '%s\n' /opt/libjpeg-turbo/lib64 > /etc/ld.so.conf.d/libjpeg-turbo-official.conf
ldconfig
/opt/libjpeg-turbo/bin/djpeg -version
