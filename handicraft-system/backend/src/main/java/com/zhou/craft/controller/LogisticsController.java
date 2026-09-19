package com.zhou.craft.controller;
import com.zhou.craft.dto.ShipForm;import com.zhou.craft.repo.CraftOrderRepository;import com.zhou.craft.security.OrderAccessService;import com.zhou.craft.service.LogisticsService;import org.springframework.web.bind.annotation.*;import java.util.*;
@RestController @RequestMapping("/api/logistics")public class LogisticsController{
private final CraftOrderRepository orders;private final LogisticsService logistics;private final OrderAccessService access;public LogisticsController(CraftOrderRepository o,LogisticsService l,OrderAccessService a){orders=o;logistics=l;access=a;}
@GetMapping("/order/{id}")public Map<String,Object> detail(@PathVariable Long id){var o=orders.findById(id).orElseThrow();access.requireView(o);var s=logistics.findByOrderId(id);Map<String,Object>m=new LinkedHashMap<>();m.put("shipment",s.orElse(null));m.put("traces",s.map(x->logistics.traces(x.getId())).orElse(List.of()));return m;}
@PostMapping("/order/{id}/ship")public Object ship(@PathVariable Long id,@RequestBody(required=false)ShipForm f){var o=orders.findById(id).orElseThrow();access.requireCreator(o);return logistics.ship(o,f);}
@PostMapping("/order/{id}/advance")public Object advance(@PathVariable Long id){var o=orders.findById(id).orElseThrow();access.requireCreator(o);return logistics.advance(o);}
}