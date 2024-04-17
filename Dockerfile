FROM python:3.11-alpine
ENV PROJECT_NAME=python-slideshow
ENV WORKDIR=/${PROJECT_NAME}
WORKDIR ${WORKDIR}
ADD . ${WORKDIR}/${PROJECT_NAME}

ARG GUI=false
ARG SERVER=false

RUN if [ "$GUI" = "true" ]; then \
        apk --update-cache add --no-cache qt6-qtbase-dev python3-pyqt6; \
        PIP_INSTALL_CMD_SUFFIX="gui,"; \
    fi \
    && if [ "$SERVER" = "true" ]; then \
        PIP_INSTALL_CMD_SUFFIX=${PIP_INSTALL_CMD_SUFFIX}"server"; \
        echo $PIP_INSTALL_CMD_SUFFIX; \
    fi \
    && apk --update-cache add --no-cache ffmpeg \ 
    && eval "pip install ${WORKDIR}/${PROJECT_NAME}[${PIP_INSTALL_CMD_SUFFIX}]" \
    && rm -r ${WORKDIR}/${PROJECT_NAME}

ENTRYPOINT ["slideshow"]
CMD [ \
        "runserver", \
        "--srcdir", "/python-slideshow/media/disk", \
        "--dstdir", "/python-slideshow/media/_temp", \
        "--port", "10011", \
        "--log-level", "INFO", \
        "--ffmpeg-loglevel", "QUIET" \
    ]
EXPOSE 10011