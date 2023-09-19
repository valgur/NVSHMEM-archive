/*
 * Copyright (c) 2016-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef SOCKETS_H
#define SOCKETS_H

#include <sys/types.h>

typedef struct ipcHandle_st {
    int socket;
    char *socketName;
} ipcHandle;

int ipcOpenSocket(ipcHandle *&handle);

int ipcCloseSocket(ipcHandle *handle);

int ipcRecvFd(ipcHandle *handle, int *fd);

int ipcSendFd(ipcHandle *handle, const int fd, pid_t process);
int ipcCloseFd(int fd);

#endif /* SOCKETS_H */
