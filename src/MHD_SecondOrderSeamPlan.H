#pragma once
#include "Proto.H"
#include <vector>
#include <memory>
#include <algorithm>
#include <limits>

namespace HelioCubed
{
// Compiled access to an immutable stage stencil and its retained donor
// allocation. Coefficients and accumulation order match MBInterpOp::apply.
// Only the O2 donor cache owns this plan; redefining that cache destroys it.
template<typename T,unsigned C>
class SecondOrderSeamPlan
{
    using Level=Proto::MBLevelBoxData<T,C,Proto::HOST>;
    struct Source {const T* data;size_t stride;};
    struct Run
    {
        T* data;
        size_t stride,first;
        int count;
        const std::vector<double>* coefficients;
    };
    struct Storage {const T* data;Proto::Box box;};
    const Level* m_destination;
    const Level* m_source;
    const Proto::MBInterpOp* m_operator;
    std::vector<std::shared_ptr<Proto::MBPointInterpOp>> m_ops;
    std::vector<Storage> m_storage;
    std::vector<Run> m_runs;
    std::vector<Source> m_sources;

    static bool shifted(const Proto::MBDataPoint& a,const Proto::MBDataPoint& b)
    {
        const auto step=Proto::Point::Basis(0);
        return a.dstIndex()==b.dstIndex() && a.srcIndex()==b.srcIndex()
            && a.dstBlock()==b.dstBlock() && a.srcBlock()==b.srcBlock()
            && a.boundaryDir()==b.boundaryDir()
            && b.dstPoint()==a.dstPoint()+step && b.srcPoint()==a.srcPoint()+step;
    }
    static bool follows(const Proto::MBPointInterpOp& a,const Proto::MBPointInterpOp& b)
    {
        if (!shifted(a.target(),b.target()) || a.coefs()!=b.coefs()
            || a.sources().size()!=b.sources().size()) return false;
        for (size_t i=0;i<a.sources().size();++i)
            if (!shifted(a.sources()[i],b.sources()[i])) return false;
        return true;
    }
public:
    SecondOrderSeamPlan(Level& destination,Level& source,const Proto::MBInterpOp& op)
        :m_destination(&destination),m_source(&source),m_operator(&op),m_ops(op.getOps())
    {
        PR_TIMERS("HelioCubed::seamPlanSetup");
        PROTO_ASSERT(&destination!=&source,"Compiled seams require separate donor storage");
        for (auto index:destination.layout())
            m_storage.push_back({destination[index].data(),destination[index].box()});
        for (size_t first=0;first<m_ops.size();)
        {
            size_t end=first+1;
            while (end<m_ops.size() && follows(*m_ops[end-1],*m_ops[end])) ++end;
            const auto& pointOp=*m_ops[first];
            auto& dst=pointOp.target().srcPatch(destination);
            const auto target=pointOp.target().dstPoint();
            auto probe=target;probe[0]=dst.box().low()[0];
            const int low=std::max(target[0],dst.box().low()[0]);
            const int high=std::min(target[0]+int(end-first)-1,dst.box().high()[0]);
            if (dst.box().containsPoint(probe) && low<=high && !pointOp.sources().empty())
            {
                const auto shift=Proto::Point::Basis(0,low-target[0]);
                auto output=dst.var(target+shift);
                Run run{&output(0),dst.box().size(),m_sources.size(),high-low+1,&pointOp.coefs()};
                for (const auto& donor:pointOp.sources())
                {
                    auto& input=donor.srcPatch(source);
                    const auto p=donor.dstPoint()+shift;
                    PROTO_ASSERT(input.box().containsPoint(p)
                        && input.box().containsPoint(p+Proto::Point::Basis(0,run.count-1)),
                        "Compiled seam donor does not cover the radial run");
                    auto value=input.var(p);
                    m_sources.push_back({&value(0),input.box().size()});
                }
                m_runs.push_back(run);
            }
            first=end;
        }
    }
    bool matches(const Level& destination,const Level& source,const Proto::MBInterpOp& op) const
    {
        if (&destination!=m_destination || &source!=m_source || &op!=m_operator) return false;
        const auto& ops=op.getOps();
        if (ops.size()!=m_ops.size()) return false;
        for (size_t i=0;i<ops.size();++i) if (ops[i]!=m_ops[i]) return false;
        size_t i=0;
        for (auto index:destination.layout())
        {
            if (i>=m_storage.size() || destination[index].data()!=m_storage[i].data
                || destination[index].box()!=m_storage[i].box) return false;
            ++i;
        }
        return i==m_storage.size();
    }
    void apply(bool boundPrimitives=false) const
    {
        PR_TIMERS("HelioCubed::applySeamPlan");
        for (const auto& run:m_runs)
        {
            for (unsigned c=0;c<C;++c)
            {
                T* output=run.data+c*run.stride;
                for (int r=0;r<run.count;++r) output[r]=0.;
            }
            for (size_t s=0;s<run.coefficients->size();++s)
            {
                const auto& source=m_sources[run.first+s];
                const double coefficient=(*run.coefficients)[s];
                for (unsigned c=0;c<C;++c)
                {
                    T* output=run.data+c*run.stride;
                    const T* input=source.data+c*source.stride;
                    for (int r=0;r<run.count;++r) output[r]+=coefficient*input[r];
                }
            }
            if(boundPrimitives) for(int r=0;r<run.count;++r)
            {
                // Polynomial seam weights can be negative. With a sharp CME
                // density contrast, clipping a negative interpolated density
                // to a tiny floor creates an enormous ghost-cell Alfven speed.
                // Use a convex interpolation of the complete primitive state
                // when rho or p leaves its donor bounds, instead of inventing
                // a near-vacuum ghost. Owned conserved cells are never changed.
                T rhoMin=std::numeric_limits<T>::infinity(),rhoMax=0;
                T pMin=std::numeric_limits<T>::infinity(),pMax=0;
                for(size_t s=0;s<run.coefficients->size();++s) {
                    const auto& source=m_sources[run.first+s];
                    const T rho=source.data[r],p=source.data[r+4*source.stride];
                    rhoMin=std::min(rhoMin,rho);rhoMax=std::max(rhoMax,rho);
                    pMin=std::min(pMin,p);pMax=std::max(pMax,p);
                }
                const T rho=run.data[r],p=run.data[r+4*run.stride];
                if(rho>=rhoMin && rho<=rhoMax && p>=pMin && p<=pMax) continue;
                T weight=0;
                for(unsigned c=0;c<C;++c) run.data[r+c*run.stride]=0;
                for(size_t s=0;s<run.coefficients->size();++s) {
                    const T coefficient=std::max(0.,(*run.coefficients)[s]);
                    const auto& source=m_sources[run.first+s];weight+=coefficient;
                    for(unsigned c=0;c<C;++c)
                        run.data[r+c*run.stride]+=coefficient*source.data[r+c*source.stride];
                }
                PROTO_ASSERT(weight>0,"Seam interpolation needs a positive donor weight");
                for(unsigned c=0;c<C;++c) run.data[r+c*run.stride]/=weight;
            }
        }
    }
};
}
