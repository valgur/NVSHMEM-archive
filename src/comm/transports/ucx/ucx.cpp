/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "ucx.h"

static struct gdrcopy_function_table gdrcopy_ftable;
static void *gdrcopy_handle = NULL;
static gdr_t gdr_desc;

static std::deque<void *>   pending_recv_headers;
static std::deque<void *>   free_recv_headers;

static uint64_t nvshmemt_ucx_submitted_proxy_atomics = 0;
static uint64_t nvshmemt_ucx_submitted_host_atomics = 0;
static uint64_t nvshmemt_ucx_completed_proxy_atomics = 0;
static uint64_t nvshmemt_ucx_completed_host_atomics = 0;

static uint64_t nvshmemt_ucx_recv_headers_in_use = 0;
static uint64_t nvshmemt_ucx_p_buffers_in_use = 0;

int nvshmemt_ucx_progress(nvshmem_transport_t transport);

static void nvshmemt_ucx_send_request_cb(void *request, ucs_status_t status, void *user_data) {
    if (status != UCS_OK) {
        ERROR_PRINT("UCX send request completed with error.\n");
    }

    if (user_data) {
        nvshmemt_ucx_p_buffer_t *buffer;
        buffer = (nvshmemt_ucx_p_buffer_t *)user_data;
        buffer->in_use = false;
        nvshmemt_ucx_p_buffers_in_use--;
    }
    ucp_request_free(request);
    return;
}

static void nvshmemt_ucx_send_am_request_cb(void *request, ucs_status_t status, void *user_data) {
    nvshmemt_ucx_am_header_t *header_info = (nvshmemt_ucx_am_header_t *)user_data;

    if (status != UCS_OK) {
        ERROR_PRINT("UCX send request completed with error.\n");
    }

    if (header_info == NULL) {
        ERROR_PRINT("UCX send request completed with error.\n");
        return;
    }

    nvshmemt_ucx_recv_headers_in_use--;
    header_info->in_use = false;
    ucp_request_free(request);
    return;
}

static ucs_status_t nvshmemt_ucx_recv_send_am_data_cb(void *arg, const void *header,
                                                      size_t header_length, void *data,
                                                      size_t length, const ucp_am_recv_param_t *param) {
    struct nvshmem_transport    *transport = (struct nvshmem_transport *)arg;
    nvshmemt_ucx_am_header_t    *header_info;
    nvshmemt_ucx_am_header_t    *buffer_header;
    bool                        dynamic_allocation = false;
    ucs_status_t                status;

    assert(transport);
    assert(length == sizeof(nvshmemt_ucx_am_header_t));
    /* I believe this is guaranteed by the way we specify our flags. */
    assert(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));

    /* UCP_AM_RECV_ATTR_FLAG_DATA implies that we can keep the data.
     * Until https://github.com/openucx/ucx/pull/6005 is released
     * we can't guarantee this will be the case. Meanwhile, we need
     * a workaround.
     */
    if(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA)) {
        buffer_header = (nvshmemt_ucx_am_header_t *)data;
        if (!free_recv_headers.empty()) {
            header_info = (nvshmemt_ucx_am_header_t *)free_recv_headers.front();
            free_recv_headers.pop_front();
        /* Unfortunately, we also can't call progress from within
         * the callback to free up headers. So we need to allocate
         * more headers if we are out.
         * #TODO: remove when https://github.com/openucx/ucx/pull/6005
         * is merged to a release branch.
         */
        } else {
            dynamic_allocation = true;
            header_info = (nvshmemt_ucx_am_header_t *)calloc(1, sizeof(nvshmemt_ucx_am_header_t));
            if (header_info == NULL) {
                ERROR_EXIT("Unable to allocate memory in UCX transport.\n");
            }
        }
        memcpy(header_info, buffer_header, sizeof(nvshmemt_ucx_am_header_t));
        header_info->nvshmem_owned = true;
        header_info->dynamic_alloc = dynamic_allocation;
        status = UCS_OK;
    } else {
        header_info = (nvshmemt_ucx_am_header_t *)data;
        header_info->nvshmem_owned = false;
        status = UCS_INPROGRESS;
    }

    header_info->header.send_h.ep = param->reply_ep;

    pending_recv_headers.push_back((void *)header_info);

    return status;
}

ucs_status_t ucx_recv_resp_am_data_cb(void *arg, const void *header,
                                      size_t header_length, void *data,
                                      size_t length, const ucp_am_recv_param_t *param) {
    struct nvshmem_transport        *transport = (struct nvshmem_transport *)arg;
    transport_ucx_state_t           *ucx_state = (transport_ucx_state_t *)transport->state;
    nvshmemt_ucx_mem_handle_info_t  *mem_handle_info = &ucx_state->mem_handle_info;
    nvshmemt_ucx_am_header_t        *header_info;
    volatile g_elem_t               *recv_elem_ptr;
    void                            *valid_cpu_ptr;
    bool                            is_proxy;

    if (!header_length) {
        ERROR_PRINT("Got NULL header in resp_am_data cb.\n");
        goto out;
    }
    if (header_length == sizeof(nvshmemt_ucx_am_header_t)) {
        header_info = (nvshmemt_ucx_am_header_t *)header;

        valid_cpu_ptr = (void *)((char *)mem_handle_info->cpu_ptr +
                                ((char *)header_info->header.resp_h.retptr - (char *)mem_handle_info->ptr));
        recv_elem_ptr = (volatile g_elem_t *)valid_cpu_ptr;
        recv_elem_ptr->data = header_info->header.resp_h.retval;
        recv_elem_ptr->flag = header_info->header.resp_h.retflag;
        is_proxy = header_info->is_proxy;
    } else if (header_length == sizeof(bool)) {
        is_proxy = *((bool *)header);
    } else {
        ERROR_PRINT("Got bad header size (%lu) in resp_am_data cb.\n", header_length);
        goto out;
    }

    if (is_proxy) {
        nvshmemt_ucx_completed_proxy_atomics++;
    } else {
        nvshmemt_ucx_completed_host_atomics++;
    }

out:
    return UCS_OK;
}

int nvshmemt_ucx_release_mem_handle(nvshmem_mem_handle_t mem_handle, nvshmem_transport_t t) {
    nvshmemt_ucx_mem_handle_t *handle = (nvshmemt_ucx_mem_handle_t *)&mem_handle;
    transport_ucx_state_t *ucx_state = (transport_ucx_state_t *)t->state;
    nvshmemt_ucx_mem_handle_info_t *handle_info = &ucx_state->mem_handle_info;
    ucs_status_t ucs_rc;
    int status;

    ucs_rc = ucp_mem_unmap(ucx_state->library_context, handle->mem_handle);
    if (ucs_rc != UCS_OK) {
        return NVSHMEMX_ERROR_INTERNAL;
    }

    status = gdrcopy_ftable.unmap(gdr_desc, handle_info->mh, handle_info->cpu_ptr_base,
                                    handle_info->size);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr_unmap failed\n");

    status = gdrcopy_ftable.unpin_buffer(gdr_desc, handle_info->mh);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdr_unpin failed\n");

out:
    return status;
}

int nvshmemt_ucx_get_mem_handle(nvshmem_mem_handle_t *mem_handle, nvshmem_mem_handle_t mem_handle_in,
                                void *buf, size_t length, nvshmem_transport_t t) {
    ucs_status_t ucs_rc;
    transport_ucx_state_t *ucx_state = (transport_ucx_state_t *)t->state;
    nvshmemt_ucx_mem_handle_t *handle = (nvshmemt_ucx_mem_handle_t *)mem_handle;
    nvshmemt_ucx_mem_handle_info_t *handle_info = &ucx_state->mem_handle_info; 
    ucp_mem_map_params_t params;
    int status = 0;
    void *rkey = NULL;

    handle->mem_handle = NULL;
    handle_info->ptr = buf;
    handle_info->size = length;
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.flags = 0;
    params.address = buf;
    params.length = length;

    ucs_rc = ucp_mem_map(ucx_state->library_context, &params, &handle->mem_handle);
    if (ucs_rc != UCS_OK) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                  "Failed to map memory in UCX transport.\n");
    }

    ucs_rc = ucp_rkey_pack(ucx_state->library_context, handle->mem_handle, &rkey, &handle->rkey_packed_buf_len);
    if (ucs_rc != UCS_OK) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                  "Failed to pack rkey for memory region in UCX transport.\n");
    }

    status = gdrcopy_ftable.pin_buffer(gdr_desc, (unsigned long)buf, length, 0, 0, &handle_info->mh);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error, "gdrcopy pin_buffer failed \n");

    status = gdrcopy_ftable.map(gdr_desc, handle_info->mh, &handle_info->cpu_ptr_base, length);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error, "gdrcopy map failed \n");

    gdr_info_t info;
    status = gdrcopy_ftable.get_info(gdr_desc, handle_info->mh, &info);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error, "gdrcopy get_info failed \n");

    // remember that mappings start on a 64KB boundary, so let's
    // calculate the offset from the head of the mapping to the
    // beginning of the buffer
    handle_info->cpu_ptr = (void *)((char *)handle_info->cpu_ptr_base + ((char *)buf - (char *)info.va));

    /* TODO: Find a way that doesn't rely on the rkey being smaller than an arbitrary value. */
    assert(handle->rkey_packed_buf_len < NVSHMEMT_UCP_RKEY_PACKED_MAX_LEN);

    memcpy(handle->rkey_packed_buf, rkey, handle->rkey_packed_buf_len);
    ucp_rkey_buffer_release(rkey);

    return status;

error:
    if (handle->mem_handle) {
        ucp_mem_unmap(ucx_state->library_context, handle->mem_handle);
    }

    return status;
}

int nvshmemt_ucx_connect_endpoints(nvshmem_transport_t t) {
    transport_ucx_state_t *ucx_state = (transport_ucx_state_t *)t->state;
    ucx_ep_handle_t local_ep_handle, *ep_handles = NULL;
    ucs_status_t ucs_rc;
    ucp_ep_params_t params;
    ucp_address_t *local_addr;
    size_t addr_len;
    int i, j;

    int status = 0, ep_count;

    ep_count = ucx_state->ep_count = MAX_TRANSPORT_EP_COUNT + 1;
    ucx_state->proxy_ep_idx = MAX_TRANSPORT_EP_COUNT;
    
    ucx_state->endpoints = (ucp_ep_h *)calloc(nvshmemi_state->npes * ep_count, sizeof(ucp_ep_h));
    NULL_ERROR_JMP(ucx_state->endpoints, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "Failed to allocate endpoints for UCX transport.\n");

    ucx_state->ep_rkeys = (ucp_rkey_h *)calloc(nvshmemi_state->npes * ep_count, sizeof(ucp_rkey_h));
    NULL_ERROR_JMP(ucx_state->endpoints, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "Failed to allocate rkey pointers for UCX transport.\n");

    ep_handles = (ucx_ep_handle_t *)calloc(nvshmemi_state->npes, sizeof(ucx_ep_handle_t));
    NULL_ERROR_JMP(ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "Failed to allocate endpoint handles for UCX transport.\n");

    ucs_rc = ucp_worker_get_address(ucx_state->worker_context, &local_addr, &addr_len);
    if (ucs_rc != UCS_OK) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                  "Failed to get local address for endpoint in UCX transport.\n");
    }

    if(addr_len > NVSHMEMT_UCP_ADDR_MAX_LEN) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                  "Address allocated by UCX is too large. Exiting.\n");
    }
    local_ep_handle.addr_len = addr_len;
    memcpy(local_ep_handle.addr, local_addr, addr_len);
    ucp_worker_release_address(ucx_state->worker_context, local_addr);

    status = nvshmemi_state->boot_handle.allgather(&local_ep_handle, ep_handles,
                                                   sizeof(ucx_ep_handle_t),
                                                   &nvshmemi_state->boot_handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                 "Failed to gather ep_handles in UCX transport.\n");

    params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    for(i = 0; i < nvshmemi_state->npes; i++) {
        for (j = 0; j < ep_count; j++) {
            params.address = (ucp_address_t *)ep_handles[i].addr;
            ucs_rc = ucp_ep_create(ucx_state->worker_context, &params, &ucx_state->endpoints[i * ep_count + j]);
            if (ucs_rc != UCS_OK) {
                ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Failed to connect endpoint in UCX transport.\n");
            }
        }
    }

out:
    if (status != 0) {
        for(i = 0; i < nvshmemi_state->npes; i++) {
            for (j = 0; j < ep_count; j++) {
                if (ucx_state->endpoints[i * ep_count + j] != NULL) {
                    ucp_ep_close_nb(ucx_state->endpoints[i * ep_count + j], UCP_EP_CLOSE_MODE_FLUSH);
                }
            }
        }
        free(ucx_state->endpoints);
    }

    free(ep_handles);
    return status;
}

int nvshmemt_ucx_can_reach_peer(int *access, struct nvshmem_transport_pe_info *peer_info,
                                nvshmem_transport_t t) {
    *access = NVSHMEM_TRANSPORT_CAP_CPU_WRITE | NVSHMEM_TRANSPORT_CAP_CPU_READ |
              NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS;

    return 0;
}

nvshmemt_ucx_am_header_t *nvshmemt_ucx_get_header(struct nvshmem_transport *transport) {
    transport_ucx_state_t *ucx_state = (transport_ucx_state_t *)transport->state;
    int idx;

    /* Prevent using a buffer twice at the same time,
     * and prevent overflow (i.e. calling more operations than we have buffers at once).
     * Note: starvation is not a problem here since ucp_worker_progress won't invoke any
     * functions that take headers, only ones that clean it up. We intentionally don't
     * call the nvshmem transport progress function here to avoid that case.
     */
    while(nvshmemt_ucx_recv_headers_in_use > NVSHMEM_UCX_ATOMIC_POOL_MASK) {
        ucp_worker_progress(ucx_state->worker_context);
    }

    idx = __sync_fetch_and_add(&ucx_state->num_headers_requested, 1);
    idx &= NVSHMEM_UCX_ATOMIC_POOL_MASK;

    while(ucx_state->send_headers[idx].in_use) {
        ucp_worker_progress(ucx_state->worker_context);
    }

    nvshmemt_ucx_recv_headers_in_use++;
    ucx_state->send_headers[idx].in_use = true;
    return &ucx_state->send_headers[idx];
}

static nvshmemt_ucx_p_buffer_t *nvshmemt_ucx_get_p_buffer(transport_ucx_state_t *ucx_state) {
    int idx;

    while(nvshmemt_ucx_p_buffers_in_use > NVSHMEM_UCX_P_BUFFER_POOL_MASK) {
        ucp_worker_progress(ucx_state->worker_context);
    }

    idx = __sync_fetch_and_add(&ucx_state->num_p_buffers_requested, 1);
    idx &= NVSHMEM_UCX_ATOMIC_POOL_MASK;

    while(ucx_state->p_buffers[idx].in_use) {
        ucp_worker_progress(ucx_state->worker_context);
    }

    nvshmemt_ucx_p_buffers_in_use++;
    ucx_state->p_buffers[idx].in_use = true;

    return &ucx_state->p_buffers[idx];
}

int nvshmemt_ucx_rma(struct nvshmem_transport *tcurr, int pe, rma_verb_t verb, rma_memdesc_t remote, rma_memdesc_t local,
                     rma_bytesdesc_t bytesdesc, int is_proxy) {
    transport_ucx_state_t *ucx_state = (transport_ucx_state_t *)tcurr->state;
    ucp_ep_h ep;
    ucs_status_t ucs_rc;
    ucs_status_ptr_t ucs_ptr_rc = NULL;
    ucp_request_param_t param;
    nvshmemt_ucx_p_buffer_t *buffer = NULL;
    int ep_index;
    ucp_rkey_h rkey;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    param.cb.send = nvshmemt_ucx_send_request_cb;
    param.memory_type = UCS_MEMORY_TYPE_CUDA;

    if (is_proxy) {
        ep_index = (ucx_state->ep_count * pe + ucx_state->proxy_ep_idx);
    } else {
        ep_index = (ucx_state->ep_count * pe);
    }

    ep = ucx_state->endpoints[ep_index];

    if (unlikely(!ucx_state->ep_rkeys[ep_index])) {
        nvshmemt_ucx_mem_handle_t *mem_handle = (nvshmemt_ucx_mem_handle_t *)&remote.handle;
        ucs_rc = ucp_ep_rkey_unpack(ep, mem_handle->rkey_packed_buf, &ucx_state->ep_rkeys[ep_index]);
        if (ucs_rc != UCS_OK) {
            ERROR_EXIT("Unable to unpack rkey in UCS transport! Exiting.\n");
        }
    }

    rkey = ucx_state->ep_rkeys[ep_index];

    /* We aren't worried about completion, correct? */
    if (verb.desc == NVSHMEMI_OP_P) {
        buffer = nvshmemt_ucx_get_p_buffer(ucx_state);
        param.op_attr_mask |= UCP_OP_ATTR_FIELD_USER_DATA;
        param.user_data = buffer;
        switch(bytesdesc.elembytes * bytesdesc.nelems) {
            case 1:
                buffer->buffer.buffer_1_byte = *(uint8_t *)local.ptr;
                local.ptr = &buffer->buffer.buffer_1_byte;
                break;
            case 2:
                buffer->buffer.buffer_2_byte = *(uint16_t *)local.ptr;
                local.ptr = &buffer->buffer.buffer_2_byte;
                break;
            case 4:
                buffer->buffer.buffer_4_byte = *(uint32_t *)local.ptr;
                local.ptr = &buffer->buffer.buffer_4_byte;
                break;
            case 8:
                buffer->buffer.buffer_8_byte = *(uint64_t *)local.ptr;
                local.ptr = &buffer->buffer.buffer_8_byte;
                break;
            default:
                ERROR_PRINT("Received an invalid size for p operation.\n");
                nvshmemt_ucx_p_buffers_in_use--;
                buffer->in_use = false;
                return NVSHMEMX_ERROR_INTERNAL;
        }
        ucs_ptr_rc = ucp_put_nbx(ep, local.ptr,
                                bytesdesc.elembytes * bytesdesc.nelems,
                                (uint64_t)remote.ptr, rkey, &param);
    } else if (verb.desc == NVSHMEMI_OP_PUT) {
        ucs_ptr_rc = ucp_put_nbx(ep, local.ptr,
                                bytesdesc.elembytes * bytesdesc.nelems,
                                (uint64_t)remote.ptr, rkey, &param);
    } else if (verb.desc == NVSHMEMI_OP_G || verb.desc == NVSHMEMI_OP_GET) {
        ucs_ptr_rc = ucp_get_nbx(ep, local.ptr,
                                 bytesdesc.elembytes * bytesdesc.nelems,
                                 (uint64_t)remote.ptr, rkey, &param);
    }

    if (ucs_ptr_rc != NULL) {
        if (UCS_PTR_IS_ERR(ucs_ptr_rc)) {
            if (buffer) {
                nvshmemt_ucx_p_buffers_in_use--;
                buffer->in_use = false;
            }
            ERROR_PRINT("Failed in UCX Transport during RMA operation.\n");
            return NVSHMEMX_ERROR_INTERNAL;
        }
    } else {
        if (buffer) {
            nvshmemt_ucx_p_buffers_in_use--;
            buffer->in_use = false;
        }
    }

    return 0;
}

template <typename T>
int nvshmemt_ucx_handle_amo(struct nvshmem_transport *transport, ucp_ep_h ep,
                            nvshmemi_amo_t op, void *ptr, uint64_t swap_add,
                            uint64_t compare, void *retptr, uint64_t retflag, bool is_proxy) {
    T old_value, new_value;
    nvshmemt_ucx_am_header_t *header = NULL;
    ucs_status_ptr_t ucs_rc;
    ucp_request_param_t param;
    int status = 0;
    bool send_full_header = false;

    old_value = *((volatile T *)ptr);
    switch(op) {
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SET:
        case NVSHMEMI_AMO_SWAP: {
            /* The static_cast is used to truncate the uint64_t value of swap_add back to its original length */
            new_value = static_cast<T>(swap_add);
            break;
        }
        case NVSHMEMI_AMO_ADD:
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_FETCH_ADD: {
            new_value = old_value + static_cast<T>(swap_add);
            break;
        }
        case NVSHMEMI_AMO_OR:
        case NVSHMEMI_AMO_FETCH_OR: {
            new_value = old_value | static_cast<T>(swap_add);
            break;
        }
        case NVSHMEMI_AMO_AND:
        case NVSHMEMI_AMO_FETCH_AND: {
            new_value = old_value & static_cast<T>(swap_add);
            break;
        }
        case NVSHMEMI_AMO_XOR:
        case NVSHMEMI_AMO_FETCH_XOR: {
            new_value = old_value ^ static_cast<T>(swap_add);
            break;
        }
        case NVSHMEMI_AMO_COMPARE_SWAP: {
            new_value = (old_value == static_cast<T>(compare)) ? static_cast<T>(swap_add) : old_value;
            break;
        }
        default: {
            ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "RMA/AMO verb %d not implemented\n", op);
        }
    }

    *((volatile T *)ptr) = new_value;
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_USER_DATA;
    param.cb.send = nvshmemt_ucx_send_am_request_cb;
    header = nvshmemt_ucx_get_header(transport);
    param.user_data = header;
    header->is_proxy = is_proxy;
    if (op > NVSHMEMI_AMO_END_OF_NONFETCH) {
        header->header.resp_h.retptr = retptr;
        header->header.resp_h.retval = old_value;
        header->header.resp_h.retflag = retflag;
        send_full_header = true;
    }
    if (send_full_header) {
        ucs_rc = ucp_am_send_nbx(ep, NVSHMEM_UCX_ATOMIC_RESP, (void *)header,
                                 sizeof(nvshmemt_ucx_am_header_t), NULL, 0, &param);
    } else {
        ucs_rc = ucp_am_send_nbx(ep, NVSHMEM_UCX_ATOMIC_RESP, &header->is_proxy,
                                 sizeof(bool), NULL, 0, &param);
    }
    if (ucs_rc != NULL) {
        if (UCS_PTR_IS_ERR(ucs_rc)) {
            nvshmemt_ucx_recv_headers_in_use--;
            header->in_use = false;
            ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "Failed in UCX Transport during AMO.\n");
        }
    /* If ucp_am_send_nbx returns NULL, then we won't get a CB and we need to free the header now. */
    } else {
        nvshmemt_ucx_recv_headers_in_use--;
        header->in_use = false;
    }

out:
    return status;
}

int nvshmemt_ucx_process_amos(struct nvshmem_transport *transport) {
    transport_ucx_state_t *ucx_state = (transport_ucx_state_t *)transport->state;
    nvshmemt_ucx_am_header_t *header;
    nvshmemt_ucx_am_send_header_t *send_header;
    nvshmemt_ucx_mem_handle_info_t *mem_handle_info = &ucx_state->mem_handle_info;
    void *ptr;
    int status;

    while(!pending_recv_headers.empty()) {
        header = (nvshmemt_ucx_am_header_t *)pending_recv_headers.front();
        send_header = &header->header.send_h;
        ptr = (void *)((char *)mem_handle_info->cpu_ptr + ((char *)send_header->addr - (char *)mem_handle_info->ptr));
        status = 0;

        switch(send_header->op_size) {
            case 2:
                status = nvshmemt_ucx_handle_amo<uint16_t>(transport, send_header->ep,
                                                           send_header->op, ptr,
                                                           send_header->value, send_header->cmp,
                                                           send_header->retptr, send_header->retflag,
                                                           header->is_proxy);
                break;
            case 4:
                status = nvshmemt_ucx_handle_amo<uint32_t>(transport, send_header->ep,
                                                           send_header->op, ptr,
                                                           send_header->value, send_header->cmp,
                                                           send_header->retptr, send_header->retflag,
                                                           header->is_proxy);
                break;
            case 8:
                status = nvshmemt_ucx_handle_amo<uint64_t>(transport, send_header->ep,
                                                           send_header->op, ptr,
                                                           send_header->value, send_header->cmp,
                                                           send_header->retptr, send_header->retflag,
                                                           header->is_proxy);
                break;
            default:
                ERROR_PRINT("UCX bad size supplied for atomic.\n");
                status = NVSHMEMX_ERROR_INTERNAL;
        }
        pending_recv_headers.pop_front();
        /* Keep our headers around until we destroy the transport. */
        if (header->nvshmem_owned) {
            free_recv_headers.push_back((void *)header);
        } else {
            ucp_am_data_release(ucx_state->worker_context, (void *)header);
        }
        if (status != 0) {
            ERROR_PRINT("Failed to respond to atomic op in UCX.\n");
            return status;
        }
    }
    return 0;
}

int nvshmemt_ucx_amo(struct nvshmem_transport *transport, int pe, void *curetptr, amo_verb_t verb,
                     amo_memdesc_t remote, amo_bytesdesc_t bytesdesc, int is_proxy) {
    transport_ucx_state_t *ucx_state = (transport_ucx_state_t *)transport->state;
    ucs_status_ptr_t ucs_rc;
    ucp_request_param_t param;
    nvshmemt_ucx_am_header_t *header;
    ucp_ep_h ep;

    if (is_proxy) {
        ep = ucx_state->endpoints[(ucx_state->ep_count * pe + ucx_state->proxy_ep_idx)];
    } else {
        ep = ucx_state->endpoints[(ucx_state->ep_count * pe)];
    }

    header = nvshmemt_ucx_get_header(transport);
    header->header.send_h.addr = remote.ptr;
    header->header.send_h.op_size = bytesdesc.elembytes;
    header->header.send_h.value = remote.val;
    header->header.send_h.cmp = remote.cmp;
    header->header.send_h.retptr = remote.retptr;
    header->header.send_h.retflag = remote.retflag;
    header->header.send_h.op = verb.desc;
    header->is_proxy = is_proxy;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                         UCP_OP_ATTR_FIELD_USER_DATA |
                         UCP_OP_ATTR_FIELD_FLAGS;
    param.cb.send = nvshmemt_ucx_send_am_request_cb;
    param.flags = UCP_AM_SEND_FLAG_REPLY | UCP_AM_SEND_FLAG_EAGER;
    param.user_data = header;

    ucs_rc = ucp_am_send_nbx(ep, NVSHMEM_UCX_ATOMIC_SEND, NULL, 0,
                             (void *)header, sizeof(nvshmemt_ucx_am_header_t), &param);
    if (ucs_rc != NULL) {
        if (UCS_PTR_IS_ERR(ucs_rc)) {
            ERROR_PRINT("Failed in UCX Transport during AMO.\n");
            header->in_use = false;
            nvshmemt_ucx_recv_headers_in_use--;
            return NVSHMEMX_ERROR_INTERNAL;
        }
    /* If ucp_am_send_nbx returns NULL, then we won't get a CB and we need to free the header now. */
    } else {
        header->in_use = false;
        nvshmemt_ucx_recv_headers_in_use--;
    }

    if (is_proxy) {
        nvshmemt_ucx_submitted_proxy_atomics++;
    } else {
        nvshmemt_ucx_submitted_host_atomics++;
    }
    return 0;
}

int nvshmemt_ucx_fence(struct nvshmem_transport *tcurr, int pe, int is_proxy) {
    transport_ucx_state_t *ucx_state = (transport_ucx_state_t *)tcurr->state;
    ucs_status_t ucs_rc;

    ucs_rc = ucp_worker_fence(ucx_state->worker_context);
    if (ucs_rc != UCS_OK) {
        return NVSHMEMX_ERROR_INTERNAL;
    }

    return 0;
}

int nvshmemt_ucx_quiet(struct nvshmem_transport *tcurr, int pe, int is_proxy) {
    transport_ucx_state_t *ucx_state = (transport_ucx_state_t *)tcurr->state;
    ucp_request_param_t param;
    void *ucs_status;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK;
    param.cb.send = nvshmemt_ucx_send_request_cb;

    /* Since atomics are managed by a two-part request, we need to track them seperately. */
    if (is_proxy) {
        while (nvshmemt_ucx_submitted_proxy_atomics > nvshmemt_ucx_completed_proxy_atomics) {
            nvshmemt_ucx_progress(tcurr);
        }
    } else {
        while (nvshmemt_ucx_submitted_host_atomics > nvshmemt_ucx_completed_host_atomics) {
            nvshmemt_ucx_progress(tcurr);
        }
    }

    ucs_status = ucp_worker_flush_nbx(ucx_state->worker_context, &param);
    if (ucs_status != NULL) {
        if (UCS_PTR_IS_ERR(ucs_status)) {
            ERROR_PRINT("Failed in UCX Transport during quiet.\n");
            return NVSHMEMX_ERROR_INTERNAL;
        } else {
            ucs_status_t ucs_rc;
            do {
                ucp_worker_progress(ucx_state->worker_context);
                ucs_rc = ucp_request_check_status(ucs_status);
            } while (ucs_rc == UCS_INPROGRESS);
            if (ucs_rc != UCS_OK) {
                ERROR_PRINT("Failed in UCX Transport during quiet.\n");
                return NVSHMEMX_ERROR_INTERNAL;
            }
            /* request handle is freed in the callback. */
        }
    }
    
    return 0;
}

int nvshmemt_ucx_progress(nvshmem_transport_t transport) {
    transport_ucx_state_t *ucx_state = (transport_ucx_state_t *)transport->state;

    ucp_worker_progress(ucx_state->worker_context);
    nvshmemt_ucx_process_amos(transport);

    return 0;
}

int nvshmemt_ucx_finalize(nvshmem_transport_t transport) {
    transport_ucx_state_t *ucx_state = (transport_ucx_state_t *)transport->state;
    nvshmemt_ucx_am_header_t *recv_header;
    int i, j, ep_count;

    free(transport);

    while(!free_recv_headers.empty()) {
        recv_header = (nvshmemt_ucx_am_header_t *)free_recv_headers.front();
        assert(recv_header->nvshmem_owned);
        free_recv_headers.pop_front();
        if (recv_header->dynamic_alloc) {
            free(recv_header);
        }
    }

    if (!pending_recv_headers.empty()) {
        ERROR_PRINT("Discovered uncompleted active messages during UCX transport shutdown.\n");
        while(!pending_recv_headers.empty()) {
            recv_header = (nvshmemt_ucx_am_header_t *)pending_recv_headers.front();
            free_recv_headers.pop_front();

            if (recv_header->nvshmem_owned) {
                if (recv_header->dynamic_alloc) {
                    free(recv_header);
                }
            } else {
                ucp_am_data_release(ucx_state->worker_context, (void *)recv_header);
            }
        }
    }

    if (ucx_state) {
        ep_count = ucx_state->ep_count;
        for(i = 0; i < nvshmemi_state->npes; i++) {
            for (j = 0; j < ep_count; j++) {
                if (ucx_state->endpoints[i * ep_count + j] != NULL) {
                    ucp_ep_close_nb(ucx_state->endpoints[i * ep_count + j], UCP_EP_CLOSE_MODE_FLUSH);
                }
                if (ucx_state->ep_rkeys[i * ep_count + j] != NULL) {
                    ucp_rkey_destroy(ucx_state->ep_rkeys[i * ep_count + j]);
                }
            }
        }

        free(ucx_state->endpoints);

        if (ucx_state->worker_context) {
            ucp_worker_destroy(ucx_state->worker_context);
        }

        if (ucx_state->library_context) {
            ucp_cleanup(ucx_state->library_context);
        }

        if (ucx_state->library_config) {
            ucp_config_release(ucx_state->library_config);
        }

        free(ucx_state);
    }

    if (gdr_desc) {
        gdrcopy_ftable.close(gdr_desc);
    }

    if (gdrcopy_handle) {
        dlclose(gdrcopy_handle);
    }

    return 0;
}

int nvshmemt_ucx_enforce_cst_at_target(struct nvshmem_transport *tcurr) {
    int status = 0;
    transport_ucx_state_t *ucx_state = (transport_ucx_state_t *)tcurr->state;
    nvshmemt_ucx_mem_handle_info_t *mem_handle_info = &ucx_state->mem_handle_info;

    if (!mem_handle_info) return status;

    int temp;
    gdrcopy_ftable.copy_from_mapping(mem_handle_info->mh, &temp, mem_handle_info->cpu_ptr,
                                        sizeof(int));
    return status;
}

int nvshmemt_ucx_show_info(nvshmem_mem_handle_t *mem_handles, int transport_id,
                            int transport_count, int npes, int mype) {
    INFO(NVSHMEM_TRANSPORT, "UCX show info not implemented");
    return 0;
}

#define LOAD_SYM(handle, symbol, funcptr)  \
    do {                                   \
        void **cast = (void **)&funcptr;   \
        void *tmp = dlsym(handle, symbol); \
        *cast = tmp;                       \
    } while (0)

int nvshmemt_ucx_init(nvshmem_transport_t *t) {
    ucs_status_t ucs_rc;
    ucp_params_t params;

    ucp_worker_params_t worker_params;
    ucp_worker_attr_t worker_attr;

    ucp_am_handler_param_t am_param;

    nvshmem_transport_t transport = NULL;
    transport_ucx_state_t *ucx_state = NULL;

    int status = 0;
    int transport_skipped;

    transport_skipped = strncasecmp(nvshmemi_options.REMOTE_TRANSPORT,
                                        UCX_TRANSPORT_STRING,
                                        TRANSPORT_STRING_MAX_LENGTH);
    if (transport_skipped) {
        INFO(NVSHMEM_INIT, "UCX disabled by user through environment "
                            "in favor of the %s transport.\n", nvshmemi_options.REMOTE_TRANSPORT);
        status = NVSHMEMI_ERROR_SKIPPED;
        return status;
    }

    gdrcopy_handle = dlopen("libgdrapi.so.2", RTLD_LAZY);
    if (!gdrcopy_handle) {
        INFO(NVSHMEM_INIT, "UCX disabled due to missing GDRCopy library.\n");
        status = NVSHMEMI_ERROR_SKIPPED;
        return status;
    }

    LOAD_SYM(gdrcopy_handle, "gdr_runtime_get_version", gdrcopy_ftable.runtime_get_version);
    if (!gdrcopy_ftable.runtime_get_version) {
        dlclose(gdrcopy_handle);
        INFO(NVSHMEM_INIT, "UCX disabled due to GDRCopy runtime version mismatch.\n");
        status = NVSHMEMI_ERROR_SKIPPED;
        return status;
    }
    LOAD_SYM(gdrcopy_handle, "gdr_runtime_get_version", gdrcopy_ftable.driver_get_version);
    LOAD_SYM(gdrcopy_handle, "gdr_open", gdrcopy_ftable.open);
    LOAD_SYM(gdrcopy_handle, "gdr_close", gdrcopy_ftable.close);
    LOAD_SYM(gdrcopy_handle, "gdr_pin_buffer", gdrcopy_ftable.pin_buffer);
    LOAD_SYM(gdrcopy_handle, "gdr_unpin_buffer", gdrcopy_ftable.unpin_buffer);
    LOAD_SYM(gdrcopy_handle, "gdr_map", gdrcopy_ftable.map);
    LOAD_SYM(gdrcopy_handle, "gdr_unmap", gdrcopy_ftable.unmap);
    LOAD_SYM(gdrcopy_handle, "gdr_get_info", gdrcopy_ftable.get_info);
    LOAD_SYM(gdrcopy_handle, "gdr_copy_from_mapping", gdrcopy_ftable.copy_from_mapping);
    LOAD_SYM(gdrcopy_handle, "gdr_copy_to_mapping", gdrcopy_ftable.copy_to_mapping);

    gdr_desc = gdrcopy_ftable.open();
    if (!gdr_desc) {
        dlclose(gdrcopy_handle);
        INFO(NVSHMEM_INIT, "UCX disabled due to failure to open GDRCopy library.\n");
        status = NVSHMEMI_ERROR_SKIPPED;
        return status;
    }

    /* This environment variable is needed to enable g/get operations <= 64 bytes */
    status = setenv("UCX_RC_TX_INLINE_RESP", "0", 1);
    if (status) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                  "Failed to set UCX environment variable UCX_RC_TX_INLINE_RESP.\n");
    }

    transport = (nvshmem_transport_t)calloc(1, sizeof(struct nvshmem_transport));
    NULL_ERROR_JMP(transport, status, NVSHMEMX_ERROR_INTERNAL, error,
                   "Failed to allocate transport struct for UCX.\n");

    transport->is_successfully_initialized = false;

    ucx_state = (transport_ucx_state_t *)calloc(1, sizeof(transport_ucx_state_t));
    NULL_ERROR_JMP(ucx_state, status, NVSHMEMX_ERROR_INTERNAL, error,
                   "Failed to allocate ucx_state struct for UCX.\n");

    /* Initialize the free recv headers pool. */
    for (int i = 0; i < NVSHMEM_UCX_ATOMIC_POOL_SIZE; i++) {
        ucx_state->recv_headers[i].nvshmem_owned = true;
        free_recv_headers.push_back((void *)&ucx_state->recv_headers[i]);
    }
    
    ucs_rc = ucp_config_read(NULL, NULL, &ucx_state->library_config);
    if (ucs_rc != UCS_OK) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                  "Failed to read UCP configuration for UCX.\n");
    }

    ucs_rc = ucp_config_modify(ucx_state->library_config, "TLS", "ib");
    if (ucs_rc != UCS_OK) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                  "Failed to modify configuration for UCX.\n");
    }

    ucs_rc = ucp_config_modify(ucx_state->library_config, "ZCOPY_THRESH", "0");
    if (ucs_rc != UCS_OK) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                  "Failed to modify configuration for UCX.\n");
    }

    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features = UCP_FEATURE_RMA | UCP_FEATURE_AM;
    ucs_rc = ucp_init(&params, ucx_state->library_config, &ucx_state->library_context);
    if (ucs_rc != UCS_OK) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                  "Failed to initialize UCP for UCX.\n");
    }

    /* The regular worker thread needs to operate in multi mode because it has to
     * be progressed from the proxy thread while submitting work from the regular
     * CPU threads.
     */
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    ucs_rc = ucp_worker_create(ucx_state->library_context, &worker_params, &ucx_state->worker_context);
    if (ucs_rc != UCS_OK) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                  "Failed to initialize UCP worker for UCX.\n");
    }

    worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER;
    ucp_worker_query(ucx_state->worker_context, &worker_attr);
    if (ucs_rc != UCS_OK) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                  "Failed to get worker params for UCX.\n");
    }

    if (worker_attr.max_am_header < sizeof(nvshmemt_ucx_am_header_t)) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                  "Worker am header is too small in UCX transport.\n");
    }

    am_param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                          UCP_AM_HANDLER_PARAM_FIELD_CB |
                          UCP_AM_HANDLER_PARAM_FIELD_ARG;

    /* 
     * Set three callbacks for active messages.
     * One for sent atomics,
     * another for the responses to atomics,
     * and a third for gets that are too small and require bcopy.
     */
    am_param.cb = nvshmemt_ucx_recv_send_am_data_cb;
    am_param.id = NVSHMEM_UCX_ATOMIC_SEND;
    am_param.arg = transport;
    ucs_rc = ucp_worker_set_am_recv_handler(ucx_state->worker_context, &am_param);
    if (ucs_rc != UCS_OK) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                  "Failed to initialize UCP worker active message for UCX.\n");
    }

    am_param.cb = ucx_recv_resp_am_data_cb;
    am_param.id = NVSHMEM_UCX_ATOMIC_RESP;
    ucs_rc = ucp_worker_set_am_recv_handler(ucx_state->worker_context, &am_param);
    if (ucs_rc != UCS_OK) {
        ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error,
                  "Failed to initialize UCP worker active message for UCX.\n");
    }

    transport->host_ops.can_reach_peer = nvshmemt_ucx_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_ucx_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_ucx_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_ucx_release_mem_handle;
    transport->host_ops.rma = nvshmemt_ucx_rma;
    transport->host_ops.amo = nvshmemt_ucx_amo;
    transport->host_ops.fence = nvshmemt_ucx_fence;
    transport->host_ops.quiet = nvshmemt_ucx_quiet;
    transport->host_ops.finalize = nvshmemt_ucx_finalize;
    transport->host_ops.show_info = nvshmemt_ucx_show_info;
    transport->host_ops.progress = nvshmemt_ucx_progress;
    transport->host_ops.enforce_cst = nvshmemt_ucx_enforce_cst_at_target;
#ifndef NVSHMEM_PPC64LE
    transport->host_ops.enforce_cst_at_target = nvshmemt_ucx_enforce_cst_at_target;
#endif
    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    transport->state = (void *)ucx_state;
    transport->is_successfully_initialized = true;

    *t = transport;

    return status;

error:
    if (transport) {
        free(transport);
    }

    if (ucx_state) {
        if (ucx_state->worker_context) {
            ucp_worker_destroy(ucx_state->worker_context);
        }

        if (ucx_state->library_context) {
            ucp_cleanup(ucx_state->library_context);
        }

        if (ucx_state->library_config) {
            ucp_config_release(ucx_state->library_config);
        }

        free(ucx_state);
    }


    return status;
}
