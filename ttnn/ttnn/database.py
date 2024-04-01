# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import dataclasses
import sqlite3
import shutil
from typing import Optional

from loguru import logger

import ttnn

SQLITE_CONNECTION = None


@dataclasses.dataclass
class Device:
    device_id: int
    num_y_cores: int
    num_x_cores: int
    num_y_compute_cores: int
    num_x_compute_cores: int
    worker_l1_size: int
    l1_num_banks: int
    l1_bank_size: int
    address_at_first_l1_bank: int
    address_at_first_l1_cb_buffer: int
    num_banks_per_storage_core: int
    num_compute_cores: int
    num_storage_cores: int
    total_l1_memory: int
    total_l1_for_tensors: int
    total_l1_for_interleaved_buffers: int
    total_l1_for_sharded_buffers: int
    cb_limit: int


@dataclasses.dataclass
class Operation:
    operation_id: int
    name: str
    duration: float
    matches_golden: Optional[bool]
    desired_pcc: Optional[float]
    actual_pcc: Optional[float]


@dataclasses.dataclass
class Buffer:
    operation_id: int
    device_id: int
    address: int
    max_size_per_bank: int
    buffer_type: int


@dataclasses.dataclass
class BufferPage:
    operation_id: int
    device_id: int
    address: int
    core_y: int
    core_x: int
    bank_id: int
    page_index: int
    page_address: int
    page_size: int
    buffer_type: int


def delete_reports():
    global SQLITE_CONNECTION
    if SQLITE_CONNECTION is not None:
        SQLITE_CONNECTION.close()
    logger.info(f"Deleting reports from {ttnn.CONFIG.reports_path} and closing the sqlite connection.")
    shutil.rmtree(ttnn.CONFIG.reports_path, ignore_errors=True)
    SQLITE_CONNECTION = None


def get_or_create_sqlite_db():
    global SQLITE_CONNECTION

    if SQLITE_CONNECTION is not None:
        return SQLITE_CONNECTION

    delete_reports()
    logger.info(
        f"Creating reports path at {ttnn.CONFIG.reports_path} and sqlite database at {ttnn.CONFIG.sqlite_db_path}."
    )
    ttnn.CONFIG.reports_path.mkdir(parents=True, exist_ok=True)
    SQLITE_CONNECTION = sqlite3.connect(ttnn.CONFIG.sqlite_db_path)

    cursor = SQLITE_CONNECTION.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS operations
                (operation_id int, name text, duration float, matches_golden int, desired_pcc float, actual_pcc float)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS buffers
                (operation_id int, device_id int, address int, max_size_per_bank int, buffer_type int)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS buffer_pages
                (operation_id int, device_id int, address int, core_y int, core_x int, bank_id int, page_index int, page_address int, page_size int, buffer_type int)"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS devices
                (
                    device_id int,
                    num_y_cores int,
                    num_x_cores int,
                    num_y_compute_cores int,
                    num_x_compute_cores int,
                    worker_l1_size int,
                    l1_num_banks int,
                    l1_bank_size int,
                    address_at_first_l1_bank int,
                    address_at_first_l1_cb_buffer int,
                    num_banks_per_storage_core int,
                    num_compute_cores int,
                    num_storage_cores int,
                    total_l1_memory int,
                    total_l1_for_tensors int,
                    total_l1_for_interleaved_buffers int,
                    total_l1_for_sharded_buffers int,
                    cb_limit int
                )"""
    )
    SQLITE_CONNECTION.commit()
    return SQLITE_CONNECTION


DEVICE_IDS_IN_DATABASE = set()


def insert_devices(devices):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db()
    cursor = sqlite_connection.cursor()

    for device in devices:
        if device.id() in DEVICE_IDS_IN_DATABASE:
            continue
        device_info = ttnn._ttnn.reports.get_device_info(device)
        cursor.execute(
            f"""INSERT INTO devices VALUES (
                {device.id()},
                {device_info.num_y_cores},
                {device_info.num_x_cores},
                {device_info.num_y_compute_cores},
                {device_info.num_x_compute_cores},
                {device_info.worker_l1_size},
                {device_info.l1_num_banks},
                {device_info.l1_bank_size},
                {device_info.address_at_first_l1_bank},
                {device_info.address_at_first_l1_cb_buffer},
                {device_info.num_banks_per_storage_core},
                {device_info.num_compute_cores},
                {device_info.num_storage_cores},
                {device_info.total_l1_memory},
                {device_info.total_l1_for_tensors},
                {device_info.total_l1_for_interleaved_buffers},
                {device_info.total_l1_for_sharded_buffers},
                {device_info.cb_limit}
            )"""
        )
        sqlite_connection.commit()
        DEVICE_IDS_IN_DATABASE.add(device.id())


def optional_value(value):
    if value is None:
        return "NULL"
    return value


def insert_operation(operation, operation_id, duration, matches_golden, desired_pcc, actual_pcc):
    sqlite_connection = ttnn.database.get_or_create_sqlite_db()
    cursor = sqlite_connection.cursor()

    cursor.execute(
        f"INSERT INTO operations VALUES ({operation_id}, '{operation.name}', {duration}, {optional_value(matches_golden)}, {optional_value(desired_pcc)}, {optional_value(actual_pcc)})"
    )
    sqlite_connection.commit()

    if ttnn.CONFIG.enable_buffer_report:
        for buffer in ttnn._ttnn.reports.get_buffers():
            cursor.execute(
                f"""INSERT INTO buffers VALUES (
                    {operation_id},
                    {buffer.device_id},
                    {buffer.address},
                    {buffer.max_size_per_bank},
                    {buffer.buffer_type.value}
                )"""
            )
        for buffer_page in ttnn._ttnn.reports.get_buffer_pages():
            cursor.execute(
                f"""INSERT INTO buffer_pages VALUES (
                    {operation_id},
                    {buffer_page.device_id},
                    {buffer_page.address},
                    {buffer_page.core_y},
                    {buffer_page.core_x},
                    {buffer_page.bank_id},
                    {buffer_page.page_index},
                    {buffer_page.page_address},
                    {buffer_page.page_size},
                    {buffer_page.buffer_type.value}
                )"""
            )
        sqlite_connection.commit()
