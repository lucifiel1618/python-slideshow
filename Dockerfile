FROM python:3.11-alpine
ENV PROJECT_NAME=python-slideshow
ENV WORKDIR=/${PROJECT_NAME}
WORKDIR ${WORKDIR}
ADD . ${WORKDIR}/${PROJECT_NAME}

ARG GUI=false

RUN if [ "$GUI" = "true" ]; then \
        apk --update-cache add --no-cache qt6-qtbase; \
        PIP_INSTALL_CMD_SUFFIX="[gui]"; \
    fi \
    && apk --update-cache add --no-cache ffmpeg \ 
    && eval "pip install ${WORKDIR}/${PROJECT_NAME}${PIP_INSTALL_CMD_SUFFIX}" \
    && rm -r ${WORKDIR}/${PROJECT_NAME}

ENTRYPOINT ["sh"]