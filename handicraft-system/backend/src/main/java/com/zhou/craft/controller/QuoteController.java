package com.zhou.craft.controller;

import com.zhou.craft.dto.QuoteForm;
import com.zhou.craft.model.Quote;
import com.zhou.craft.repo.CustomRequestRepository;
import com.zhou.craft.repo.QuoteRepository;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/quotes")
public class QuoteController {
    private final QuoteRepository quotes;
    private final CustomRequestRepository requests;
    public QuoteController(QuoteRepository quotes,CustomRequestRepository requests){this.quotes=quotes;this.requests=requests;}

    @GetMapping
    public List<Quote> list(@RequestParam Long requestId){return quotes.findByRequestIdOrderByPriceAsc(requestId);}

    @PostMapping
    public Quote create(@RequestBody QuoteForm f){
        var q=quotes.save(new Quote(f.requestId(),f.creatorName(),f.price(),f.leadDays(),f.note()));
        requests.findById(f.requestId()).ifPresent(r->{r.setStatus("已有报价");requests.save(r);});
        return q;
    }
}
