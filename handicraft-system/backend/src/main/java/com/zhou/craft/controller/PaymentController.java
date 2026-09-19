package com.zhou.craft.controller;
import com.zhou.craft.dto.*;import com.zhou.craft.model.PaymentRecord;import com.zhou.craft.repo.CraftOrderRepository;import com.zhou.craft.security.OrderAccessService;import com.zhou.craft.service.PaymentService;import org.springframework.web.bind.annotation.*;import java.util.*;
@RestController @RequestMapping("/api/payments")public class PaymentController{
private final CraftOrderRepository orders;private final PaymentService payments;private final OrderAccessService access;public PaymentController(CraftOrderRepository o,PaymentService p,OrderAccessService a){orders=o;payments=p;access=a;}
@GetMapping("/order/{id}")public Object detail(@PathVariable Long id){var o=orders.findById(id).orElseThrow();access.requireView(o);return payments.findByOrderId(id).<Object>map(x->x).orElse(Map.of());}
@PostMapping("/order/{id}/pay")public PaymentRecord pay(@PathVariable Long id,@RequestBody(required=false)PayForm f){var o=orders.findById(id).orElseThrow();access.requireCustomer(o);return payments.pay(o,f==null?null:f.provider());}
@PostMapping("/order/{id}/refund")public PaymentRecord refund(@PathVariable Long id,@RequestBody(required=false)RefundForm f){var o=orders.findById(id).orElseThrow();access.requireRole("ADMIN");return payments.refund(o,f==null?null:f.reason());}
}