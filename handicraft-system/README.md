# 手工艺作品展示与个性化定制系统 V1

毕业设计题目：**基于 SpringBoot + Vue 的手工艺作品展示与个性化定制系统的设计与实现**

## 技术栈
- 后端：Spring Boot 3 / Spring Data JPA / H2（开发演示）/ MySQL（正式部署可切换）/ Maven / JDK 17
- 前端：Vue 3 / Vite / Element Plus / Axios / Vue Router / ECharts
- 角色：普通用户、创作者、管理员

## V1 核心业务
1. 作品分类展示、搜索与详情查看
2. 用户提交个性化定制需求
3. 创作者查看需求并进行报价
4. 用户选择报价后生成定制订单
5. 创作者更新制作进度
6. 用户查看订单和制作进度
7. 管理员查看用户、作品、需求、订单等运营数据

## 演示账号
- 普通用户：user / 123456
- 创作者：creator / 123456
- 管理员：admin / 123456

## IDEA 运行
### 后端
使用 IDEA 打开 `backend` 目录，运行：
`src/main/java/com/zhou/craft/CraftApplication.java`

默认端口：8080。

### 前端
在 IDEA Terminal 中：
```bash
cd frontend
npm install
npm run dev
```
访问 http://localhost:5173

默认使用 H2 内存数据库，方便首次直接启动；正式版后续切换 MySQL 8。
