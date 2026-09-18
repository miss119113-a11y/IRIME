package com.zhou.craft.repo;
import com.zhou.craft.model.Quote;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
public interface QuoteRepository extends JpaRepository<Quote,Long>{
    List<Quote> findByRequestIdOrderByPriceAsc(Long requestId);
}
