# 手工艺作品展示与个性化定制系统 V2 — IDEA 运行说明

## 1. 推荐环境

- Windows 10/11
- IntelliJ IDEA 2024/2025/2026 均可
- JDK 17
- MySQL 8.x
- Node.js 20.x（18+ 一般也可）
- Maven：可直接使用 IDEA 自带 Maven

项目目录：
- backend：Spring Boot 后端
- frontend：Vue 3 + Vite 前端
- database：MySQL 建库脚本

默认端口：
- 后端：http://localhost:8080
- 前端：http://localhost:5173
- MySQL：localhost:3306

## 2. 第一步：创建 MySQL 数据库

先确保 MySQL 8 已启动。

使用 MySQL Workbench、Navicat 或命令行执行：

CREATE DATABASE IF NOT EXISTS handicraft_system
DEFAULT CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

也可以直接执行 database/create_database.sql。

注意：只需要创建空数据库。系统第一次启动时会通过 JPA 自动创建数据表，并写入演示账号、作品、定制需求和订单数据。

## 3. 第二步：配置数据库密码

打开：

backend/src/main/resources/application-mysql.yml

默认配置是：
- 数据库：handicraft_system
- 用户名：root
- 密码：root
- 端口：3306

如果你的 MySQL root 密码不是 root，请把 password 改成你自己的 MySQL 密码。

也可以不修改文件，使用环境变量：
DB_HOST=localhost
DB_PORT=3306
DB_NAME=handicraft_system
DB_USERNAME=root
DB_PASSWORD=你的MySQL密码

## 4. 第三步：用 IDEA 启动后端

1. 打开 IntelliJ IDEA。
2. 选择 Open。
3. 选择解压后的 handicraft-system/backend 文件夹。
4. 等待 IDEA 自动识别 pom.xml 并下载 Maven 依赖。
5. 在 File -> Project Structure -> Project 中确认 Project SDK 为 JDK 17。
6. 找到：
   src/main/java/com/zhou/craft/CraftApplication.java
7. 右键 CraftApplication.java -> Run 'CraftApplication'。
8. 控制台出现类似 “Started CraftApplication” 即表示后端启动成功。

浏览器访问：
http://localhost:8080/api/works/featured

能看到 JSON 数据说明后端和 MySQL 已连接成功。

## 5. 第四步：启动 Vue 前端

在 IDEA 底部 Terminal 中执行：

cd ../frontend
npm install
npm run dev

如果你是单独打开 frontend 文件夹，则直接：

npm install
npm run dev

看到 Local: http://localhost:5173/ 后，在浏览器访问：
http://localhost:5173

## 6. 演示账号

普通用户：
- 用户名：user
- 密码：123456

创作者：
- 用户名：creator
- 密码：123456

管理员：
- 用户名：admin
- 密码：123456

## 7. 推荐演示流程

普通用户：
作品浏览 -> 作品详情 -> 收藏/评价 -> 发布定制需求 -> 查看报价 -> 生成订单 -> 模拟支付 -> 查看制作进度 -> 查看物流 -> 验收。

创作者：
登录创作者工作台 -> 查看需求 -> 提交报价 -> 更新制作进度 -> 制作完成 -> 模拟发货 -> 更新物流轨迹。

管理员：
登录管理后台 -> 查看系统统计 -> 查看订单和支付信息 -> 处理退款等管理操作。

## 8. 当前支付与物流说明

本系统用于本科毕业设计，不接入真实微信支付、支付宝、顺丰等生产接口。

代码采用可扩展服务接口：
- PaymentService -> MockPaymentService
- LogisticsService -> MockLogisticsService

因此当前运行效果是真实业务状态流转和数据库落库，但资金与物流轨迹为模拟数据。后续如果接第三方平台，只需要替换相应 Service 实现。

## 9. 常见问题

### MySQL 报 Access denied for user 'root'
说明数据库密码错误。修改 application-mysql.yml 中 password。

### Communications link failure
MySQL 服务没有启动、3306 端口不正确，或数据库连接参数错误。

### Unknown database 'handicraft_system'
先执行 database/create_database.sql 创建数据库。

### IDEA 提示 Java 版本错误
Project SDK 和 Maven Runner JDK 都改为 JDK 17。

### npm 不是内部或外部命令
先安装 Node.js 20.x，然后关闭并重新打开 IDEA。

### 8080 端口被占用
关闭占用 8080 的程序，或者临时修改 backend/src/main/resources/application.yml 的 server.port。

### 5173 端口被占用
Vite 通常会自动使用下一个可用端口；如果变成 5174，需要同步检查后端 CORS 配置。

## 10. 关闭系统

前端 Terminal 按 Ctrl+C。
后端在 IDEA Run 窗口点击红色 Stop 按钮。
MySQL 可以保持运行。
