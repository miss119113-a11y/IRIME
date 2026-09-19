package com.zhou.craft.controller;
import com.zhou.craft.repo.*;import org.springframework.web.bind.annotation.*;import java.util.*;
@RestController @RequestMapping("/api/dashboard")
public class DashboardController{
private final UserRepository users;private final CraftWorkRepository works;private final CustomRequestRepository requests;private final CraftOrderRepository orders;private final PaymentRecordRepository payments;
public DashboardController(UserRepository users,CraftWorkRepository works,CustomRequestRepository requests,CraftOrderRepository orders,PaymentRecordRepository payments){this.users=users;this.works=works;this.requests=requests;this.orders=orders;this.payments=payments;}
@GetMapping public Map<String,Object> dashboard(){
var ps=payments.findAll();
double paidAmount=ps.stream().filter(p->"已支付".equals(p.getStatus())).mapToDouble(p->p.getAmount()).sum();
double refundAmount=ps.stream().filter(p->"已退款".equals(p.getStatus())).mapToDouble(p->p.getAmount()).sum();
long pendingRefund=orders.findAll().stream().filter(o->"取消申请中".equals(o.getStatus())).count();
Map<String,Object> m=new LinkedHashMap<>();m.put("users",users.count());m.put("works",works.count());m.put("requests",requests.count());m.put("orders",orders.count());m.put("paidAmount",paidAmount);m.put("refundAmount",refundAmount);m.put("pendingRefund",pendingRefund);return m;
}}
