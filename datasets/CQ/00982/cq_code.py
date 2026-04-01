import cadquery as cq

result = (
cq.Sketch()
.segment((0.,0),(0.,2.))
.segment((2.,0))
.close()
.arc((.6,.6),0.4,0.,360.)
.assemble(tag='face')
.faces(">Z")
.circle(0.5)
)
cq.exporters.export(result, 'GT.stl')