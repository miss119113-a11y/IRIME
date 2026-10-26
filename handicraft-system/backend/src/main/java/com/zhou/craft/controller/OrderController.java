package com.zhou.craft.controller;

import com.zhou.craft.dto.ProgressForm;
import com.zhou.craft.model.CraftOrder;
import com.zhou.craft.repo.*;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/orders")
public class OrderController {
    private final CraftOrderRepository orders;
    private final CustomRequestRepository requests;
    private final QuoteRepository quotes;
    public OrderController(CraftOrderRepository orders,CustomRequestRepository requests,QuoteRepository quotes){
        this.orders=orders;this.requests=requests;this.quotes=quotes;
    }

    @GetMapping
    public List<CraftOrder> list(@RequestParam(required=false) String customerName){
        return customerName==null||customerName.isBlank()?orders.findAllByOrderByCreatedAtDesc():orders.findByCustomerNameOrderByCreatedAtDesc(customerName);
    }

    @PostMapping("/from-quote/{quoteId}")
    public CraftOrder create(@PathVariable Long quoteId){
        var q=quotes.findById(quoteId).orElseThrow();
        var r=requests.findById(q.getRequestId()).orElseThrow();
        r.setStatus("已转订单");
        requests.save(r);
        return orders.save(new CraftOrder(r.getId(),q.getId(),r.getTitle(),r.getUserName(),q.getCreatorName(),q.getPrice(),q.getLeadDays()));
    }

    @PutMapping("/{id}/progress")
    public CraftOrder progress(@PathVariable Long id,@RequestBody ProgressForm f){
        var o=orders.findById(id).orElseThrow();o.updateProgress(f.progress(),f.text());return orders.save(o);
    }

    @PutMapping("/{id}/complete")
    public CraftOrder complete(@PathVariable Long id){
        var o=orders.findById(id).orElseThrow();o.complete();return orders.save(o);
    }
}
